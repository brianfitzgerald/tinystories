import json
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.formatting.formatting import LazyBatch
from loguru import logger
from numpy import percentile
from torch import Tensor
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from unidecode import unidecode
import lightning.pytorch as pl

from model.utils import IGNORE_TOKEN_INDEX, SmDataset, ensure_directory
from synthetic_data.prompts import ENTITY_EXTRACTION_TUNING_INSTRUCTION
from synthetic_data.utils import ShareGPTConversation, dictl, ldictl


def format_squad_extractive(sample: dict) -> Tuple[str, str]:
    sample["json_schema"] = unidecode(sample["json_schema"])
    sample["context"] = unidecode(sample["context"])

    schema: dict = json.loads(sample["json_schema"])
    json_schema = {key: type(value).__name__ for key, value in schema.items()}
    input_out = f"Extract the following information using the provided schema: \t{json.dumps(json_schema)}\tand the following context: \t{sample['context']}\n"
    labels_out = json.dumps(schema)

    return input_out, labels_out


DPO_COLS_TO_TOKENIZE = ["chosen", "rejected", "prompt"]


class SquadExtractiveQADataModule(SmDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/parti"
        self.dataset_name = "roborovski/squad-extractive-qa"

    def process_samples_batch(self, samples: LazyBatch):
        inputs, labels = [], []
        for i in range(len(samples["id"])):  # type: ignore
            sample = {k: v[i] for k, v in samples.items()}
            sample_input, sample_labels = format_squad_extractive(sample)
            inputs.append(sample_input)
            labels.append(sample_labels)

        return self._tokenize(inputs, labels)


class SquadDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = f"dataset_caches/squad_len_{max_token_length}"
        self.dataset_name = "rajpurkar/squad_v2"

    def process_samples_batch(self, samples: LazyBatch):
        inputs, labels = [], []
        for i in range(len(samples["id"])):  # type: ignore
            sample = {k: v[i] for k, v in samples.items()}
            answers = sample["answers"]["text"]
            sample_input = (
                f"Question: {sample['question']}\nContext: {sample['context']}\n"
            )
            sample_labels = answers[0] if answers else "Cannot answer this question"
            inputs.append(sample_input)
            labels.append(sample_labels)

        return self._tokenize(inputs, labels)


class DollyEntityExtractionDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = f"dataset_caches/dolly_entity_extraction"
        self.dataset_name = "roborovski/dolly-entity-extraction"
        self.cpu_count = 1
        self.max_token_length = max_token_length

    def process_samples_batch(self, samples: LazyBatch):
        input_ids_out, labels_out, attention_masks_out = [], [], []
        for i in range(len(samples["context"])):  # type: ignore
            sample = {k: v[i] for k, v in samples.items()}

            conversation: ShareGPTConversation = [
                {
                    "role": "system",
                    "content": ENTITY_EXTRACTION_TUNING_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": sample["context"],
                },
                {
                    "role": "user",
                    "content": sample["json_query"],
                },
            ]

            conversation_completion = conversation + [
                {
                    "role": "assistant",
                    "content": sample["json_data"],
                }
            ]

            prompt_ids: Tensor = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="pt", add_generation_prompt=True)[0]  # type: ignore
            input_ids: Tensor = self.tokenizer.apply_chat_template(conversation_completion, tokenize=True, return_tensors="pt")[0]  # type: ignore

            user_prompt_len = prompt_ids.shape[0]
            labels = torch.tensor(
                [IGNORE_TOKEN_INDEX] * user_prompt_len
                + input_ids[user_prompt_len:].tolist()
            )
            pad_amt = self.max_token_length - labels.shape[0]
            labels = F.pad(labels, (0, pad_amt), value=self.tokenizer.pad_token_id)
            input_ids = F.pad(
                input_ids, (0, pad_amt), value=self.tokenizer.pad_token_id
            )

            labels_out.append(labels)
            input_ids_out.append(input_ids)
            attention_masks_out.append(torch.ones_like(input_ids))

        input_ids_out = torch.stack(input_ids_out)
        labels_out = torch.stack(labels_out)
        attention_masks_out = torch.stack(attention_masks_out)
        return {
            "input_ids": input_ids_out,
            "attention_mask": attention_masks_out,
            "labels": labels_out,
        }


def rec_extract_assistant_messages(messages, index=-1):
    """Recursively extract the last assistant messages from the end of the conversation."""
    if messages[index]["role"] == "assistant":
        return [messages[index]]
    else:
        return rec_extract_assistant_messages(messages, index - 1)


def create_triplets(
    example,
    default_system_message: Optional[str] = None,
):
    """Create the triplets (prompt, chosen, rejected)"""
    prompt_messages = example["chosen"][:-1]
    if example["chosen"][0]["role"] != "system" and default_system_message:
        prompt_messages.insert(0, {"role": "system", "content": default_system_message})
    chosen_messages = rec_extract_assistant_messages(example["chosen"])
    rejected_messages = rec_extract_assistant_messages(example["rejected"])

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }


class CodeContestsDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
        max_samples: Optional[int] = None,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/codecontests-dpo"
        self.dataset_name = "roborovski/codecontests-dpo"
        self.cpu_count = 1
        self.max_token_length = max_token_length
        self.max_samples = max_samples

    def load_dataset(self):
        # Load dataset and split
        dataset = Dataset.from_parquet("codecontests_dpo_v2_filtered.parquet")
        if self.max_samples:
            dataset = dataset.select(range(self.max_samples))  # type: ignore
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def process_samples_batch(self, examples: dict):
        # No need to tokenize when using DPOTrainer

        prompts = [
            f"{example['name']}\n{example['description']}"
            for example in dictl(examples)
        ]

        batch_out = {
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
            "prompt": prompts,
        }
        return batch_out


class UltraFeedbackDataModule(pl.LightningDataModule):

    def __init__(
        self,
        max_samples: Optional[int] = None,
        default_system_message: Optional[str] = None,
    ):
        self.dataset_name = "argilla/ultrafeedback-binarized-preferences-cleaned"
        self.max_samples = max_samples
        self.default_system_message = default_system_message
        self.num_workers = 1
        # Not used
        self.cache_dir = "dataset_caches/ultrafeedback"

    def setup(self, stage: Optional[str] = None):
        # TODO offline generate reference logps
        # TODO filter by p95 length, and compute max length for tokenization
        # Load dataset and split
        dataset = load_dataset(self.dataset_name)["train"]  # type: ignore
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        if self.max_samples:
            dataset = dataset.select(range(self.max_samples))  # type: ignore
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def process_samples_batch(self, examples: dict):
        out_dict = {k: [] for k in DPO_COLS_TO_TOKENIZE}
        for i in range(len(examples["prompt"])):
            example = {k: v[i] for k, v in examples.items()}
            triplets = create_triplets(example, self.default_system_message)
            for response_role in DPO_COLS_TO_TOKENIZE:
                out_dict[response_role].append(triplets[response_role])
        return out_dict


class EvolCodeAlpacaDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
        max_samples: Optional[int] = None,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/evol-codealpaca-dpo"
        self.dataset_name = "AlekseyKorshuk/evol-codealpaca-v1-dpo"
        self.max_token_length = max_token_length
        self.max_samples = max_samples

    def process_samples_batch(self, examples: dict):
        batch_out = {
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
            "prompt": examples["question"],
        }
        return batch_out
