from typing import Tuple

from transformers.tokenization_utils import PreTrainedTokenizer

from model.utils import (
    SmDataset,
)

TASK_PREFIX = "Extract the following information from the given context using the provided schema:"

def get_schema_from_result(d: dict):
    return {key: type(value).__name__ for key, value in d.items()}


def format_prompt(sample: dict) -> Tuple[str, str]:
    json_schema = get_schema_from_result(sample["json_schema"])
    input_out = f"{TASK_PREFIX}\t{sample["context"]}\n{sample["fields"]}"
    labels_out = f"{json_schema}"

    return input_out, labels_out

class SquadExtractiveQADataModule(SmDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)

        # Dataset specific parameters
        self.task_prefix = TASK_PREFIX
        self.cache_dir = "dataset_caches/parti"
        self.dataset_name = "roborovski/squad-extractive-qa"
        self.input_column, self.target_column = "context", "fields"

    def prepare_sample(self, examples: dict):
        breakpoint()
        formatted = [format_prompt(sample) for sample in examples]
        inputs = [x[0] for x in formatted]
        labels = [x[1] for x in formatted]

        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels_tokenized = self.tokenizer(
            labels,
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
