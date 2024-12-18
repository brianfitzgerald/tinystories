from typing import Tuple
from datasets.formatting.formatting import LazyBatch
import json
from unidecode import unidecode
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from loguru import logger
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import os

from transformers.tokenization_utils import PreTrainedTokenizer
from synthetic_data.utils import ShareGPTConversation, dictl, ldictl
from synthetic_data.prompts import ENTITY_EXTRACTION_TUNING_INSTRUCTION
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from tqdm import tqdm

from model.utils import SmDataset, IGNORE_TOKEN_INDEX, ensure_directory


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
        max_val_size: Optional[int] = None
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/codecontests-dpo"
        self.dataset_name = "roborovski/codecontests-dpo"
        self.cpu_count = 1
        self.max_token_length = max_token_length
        if max_val_size:
            logger.info(f"Max val size set to {max_val_size}")
            self.val_dataset = self.val_dataset[:max_val_size]

    def load_dataset(self):
        # Load dataset and split
        dataset = Dataset.from_parquet("codecontests_dpo_v2_filtered.parquet").train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def process_samples_batch(self, examples: dict):
        # No need to tokenize when using DPOTrainer

        prompts = [f"{example['name']}\n{example['description']}" for example in dictl(examples)]

        batch_out = {"chosen": examples["chosen"], "rejected": examples["rejected"], "prompt": prompts}
        return batch_out


class UltraFeedbackDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
        max_samples: Optional[int] = None,
        use_cache: bool = True,
    ):
        super().__init__(batch_size, tokenizer, max_token_length, use_cache)

        self.cache_dir = "dataset_caches/ultrafeedback"
        self.dataset_name = "argilla/ultrafeedback-binarized-preferences-cleaned"
        self.max_token_length = max_token_length
        self.max_samples = max_samples
        # TODO fix
        self.project_dir = ""
        self.default_system_message = None


    def load_dataset(self):
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
