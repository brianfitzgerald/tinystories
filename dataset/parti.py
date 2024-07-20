from datasets import load_dataset
from typing import Optional, Dict, List

from transformers.tokenization_utils import PreTrainedTokenizer

from model.utils import (
    PROMPT_EXPANSION_TASK_PREFIX,
    SmDataset,
    ensure_directory,
    SAFETY_TASK_PREFIX,
)


def generate_full_prompt(instruction: str, prompt: str) -> str:
    """
    Generates a prompt for a given example.
    """

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Input:\n{prompt}\n\n### Response:"
    )

def filter_rows(row: Dict, cols: List[str]) -> bool:
    prompt, upsampled = row["Prompt"], row["Upsampled"]
    if len(prompt) == 0 or len(upsampled) == 0:
        return False
    if "\n" in prompt or "\n" in upsampled:
        return False
    if len(upsampled.split(" ")) > 10:
        return False
    if len(upsampled) > 128:
        return False
    return True


class PromptUpsampleDataModule(SmDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        # Dataset specific parameters
        self.task_prefix = PROMPT_EXPANSION_TASK_PREFIX
        self.input_column, self.target_column = "Prompt", "Upsampled"
        self.cache_dir = "dataset_caches/parti"
        self.dataset_name = "roborovski/upsampled-prompts-parti"

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        # Load dataset and split
        # dataset = load_dataset("parquet", data_files={"train": "parti_prompts.parquet"})["train"].train_test_split(test_size=0.01)  # type: ignore
        dataset = load_dataset(self.dataset_name)["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        ensure_directory(self.cache_dir, clear=False)

        self.train_dataset = self.train_dataset.filter(
            filter_row,
            cache_file_name=f"{self.cache_dir}/training_filtered.parquet",
            num_proc=self.cpu_count,
        )
        self.val_dataset = self.val_dataset.filter(
            filter_row,
            cache_file_name=f"{self.cache_dir}/validation_filtered.parquet",
            num_proc=self.cpu_count,
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

        columns = [
            "input_ids",
            "attention_mask",
            "decoder_attention_mask",
            "labels",
        ]

        # Set format for PyTorch
        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)

    def process_samples_batch(self, examples: dict):
        inputs = [self.task_prefix + doc for doc in examples[self.input_column]]

        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels_tokenized = self.tokenizer(
            text_target=examples[self.target_column],
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "decoder_attention_mask": labels_tokenized["attention_mask"],
            "labels": labels_tokenized["input_ids"],
        }


def filter_row(row: Dict) -> bool:
    prompt, upsampled = row["Prompt"], row["Upsampled"]
    if len(prompt) == 0 or len(upsampled) == 0:
        return False
    if "\n" in prompt or "\n" in upsampled:
        return False
    if len(upsampled.split(" ")) > 10:
        return False
    if len(upsampled) > 128:
        return False
    return True


class PromptSafetyDataModule(PromptUpsampleDataModule):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        self.task_prefix = SAFETY_TASK_PREFIX
        self.input_column, self.target_column = "Prompt", "Upsampled"
        self.cache_dir = "dataset_caches/safety_workflows"
        self.dataset_name = "roborovski/safety-workflows-upsampled"

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
