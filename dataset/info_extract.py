
from transformers.tokenization_utils import PreTrainedTokenizer

from model.utils import (
    SmDataset,
)

TASK_PREFIX = "Below is a schema for information to extract from a document. Write a response that extracts the information from the document, following the provided schema:"

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
