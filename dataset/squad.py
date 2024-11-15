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

from transformers.tokenization_utils import PreTrainedTokenizer
from synthetic_data.utils import ExtractiveQARow, ShareGPTConversation
from synthetic_data.prompts import ENTITY_EXTRACTION_TUNING_INSTRUCTION

from model.utils import SmDataset, IGNORE_TOKEN_INDEX, ensure_directory


def format_squad_extractive(sample: dict) -> Tuple[str, str]:
    sample["json_schema"] = unidecode(sample["json_schema"])
    sample["context"] = unidecode(sample["context"])

    schema: dict = json.loads(sample["json_schema"])
    json_schema = {key: type(value).__name__ for key, value in schema.items()}
    input_out = f"Extract the following information using the provided schema: \t{json.dumps(json_schema)}\tand the following context: \t{sample['context']}\n"
    labels_out = json.dumps(schema)

    return input_out, labels_out


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


DEFAULT_SYSTEM_MESSAGE = "You are Dolphin, a helpful AI assistant."


def rec_extract_assistant_messages(messages, index=-1):
    """Recursively extract the last assistant messages from the end of the conversation."""
    if messages[index]["role"] == "assistant":
        return [messages[index]]
    else:
        return rec_extract_assistant_messages(messages, index - 1)


def create_triplets(
    example,
    tokenizer: PreTrainedTokenizer,
    default_system_message=DEFAULT_SYSTEM_MESSAGE,
):
    """Create the triplets (prompt, chosen, rejected)"""
    prompt_messages = example["chosen"][:-1]
    if example["chosen"][0]["role"] != "system":
        prompt_messages.insert(0, {"role": "system", "content": default_system_message})
    chosen_messages = rec_extract_assistant_messages(example["chosen"])
    rejected_messages = rec_extract_assistant_messages(example["rejected"])

    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
        "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
        "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False),
    }


class CodeContestsDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/dolly_entity_extraction"
        self.dataset_name = "roborovski/codecontests-dpo"
        self.cpu_count = 1
        self.max_token_length = max_token_length


class UltraFeedbackDataModule(SmDataset):

    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.cache_dir = "dataset_caches/ultrafeedback"
        self.dataset_name = "argilla/ultrafeedback-binarized-preferences-cleaned"
        self.cpu_count = 12
        self.max_token_length = max_token_length

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")

        # Load dataset and split
        dataset = load_dataset(self.dataset_name)["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        ensure_directory(self.cache_dir, clear=False)
        logger.info(
            f"Processing dataset for stage {stage}, workers: {self.cpu_count}, cache dir {self.cache_dir}"
        )

        self.train_dataset = self.train_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{self.cache_dir}/training.parquet",
            num_proc=self.cpu_count,
        )

        self.val_dataset = self.val_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"{self.cache_dir}/validation.parquet",
            num_proc=self.cpu_count,
        )
        # TODO offline generate reference logps

        columns = [
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_labels",
        ]

        # Set format for PyTorch
        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)

    def process_samples_batch(self, examples: dict):
        inputs, labels = {"rejected": [], "chosen": []}, {"rejected": [], "chosen": []}
        for i in range(len(examples["source"])):
            example = {k: v[i] for k, v in examples.items()}
            triplets = create_triplets(example, self.tokenizer)
            for response_role in ["chosen", "rejected"]:
                inputs[response_role].append(triplets["prompt"])
                labels[response_role].append(triplets[response_role])
        out_dict = {}
        for response_role in ["chosen", "rejected"]:
            tokenized = self._tokenize(inputs[response_role], labels[response_role])
            for key, value in tokenized.items():
                out_dict[f"{response_role}_{key}"] = value
        return out_dict
