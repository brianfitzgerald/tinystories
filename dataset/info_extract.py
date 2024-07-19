from typing import Tuple
from datasets.formatting.formatting import LazyBatch
import json
from unidecode import unidecode

from transformers.tokenization_utils import PreTrainedTokenizer

from model.utils import (
    SmDataset,
)


def format_prompt(sample: dict) -> Tuple[str, str]:
    sample["json_schema"] = unidecode(sample["json_schema"])
    sample["context"] = unidecode(sample["context"])

    schema: dict = json.loads(sample['json_schema'])
    json_schema = {key: type(value).__name__ for key, value in schema.items()}
    input_out = f"Extract the following information using the provided schema: \t{json.dumps(json_schema)}\tand the following context: \t{sample["context"]}\n"
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

        # Dataset specific parameters
        self.cache_dir = "dataset_caches/parti"
        self.dataset_name = "roborovski/squad-extractive-qa"
        self.input_column, self.target_column = "context", "fields"

    def prepare_sample(self, samples: LazyBatch):
        inputs, labels = [], []
        for i in range(len(samples['id'])): # type: ignore
            sample = {k: v[i] for k, v in samples.items()}
            sample_input, sample_labels = format_prompt(sample)
            inputs.append(sample_input)
            labels.append(sample_labels)

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
