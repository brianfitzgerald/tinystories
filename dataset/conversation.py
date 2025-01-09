from typing import List, Optional
import torch

from datasets.arrow_dataset import Dataset
from loguru import logger
from rich.text import Text
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path

from model.utils import (
    DatasetConfig,
    SmDataset,
    class_name_to_underscore,
)
from synthetic_data.utils import Conversation
import os

MASK_IDX = -100


class ConversationDataModule(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__(tokenizer, config)

        current_dir = Path().resolve().name
        self.prefix = ""
        print(current_dir == "notebooks")
        if current_dir == "notebooks":
            self.prefix = "../"
        assert self.config.input_dataset_name is not None
        input_dataset_name = self.config.input_dataset_name
        model_name = self.tokenizer.name_or_path  # type: ignore
        if os.path.isfile(self.config.input_dataset_name):
            input_dataset_name = os.path.basename(self.config.input_dataset_name)
        self.cache_dir = f"{self.prefix}dataset_caches/{class_name_to_underscore(self.__class__)}/{model_name}/{input_dataset_name}"
        logger.info(f"Cache dir: {self.cache_dir}")
        self.train_on_inputs = config.train_on_inputs

    def load_dataset(self):
        # Load dataset and split
        logger.info("Loading dataset")
        assert self.config.input_dataset_name is not None
        dataset = Dataset.from_parquet(f"{self.prefix}{self.config.input_dataset_name}")
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.custom_template = None
        if self.config.custom_chat_template is not None:
            with open(
                f"{self.prefix}chat_templates/{self.config.custom_chat_template}.jinja"
            ) as f:
                self.custom_template = f.read()

    def post_setup(self):
        self.train_dataset = self.train_dataset.remove_columns("conversation")
        self.val_dataset = self.val_dataset.remove_columns("conversation")

    def process_samples_batch_sft(self, examples: dict):
        out = self._tokenize_conversation(examples["conversation"])
        return out

    def _tokenize_conversation(self, conversation: Conversation) -> dict:
        tokenized_out: dict = self.tokenizer.apply_chat_template(
            conversation,  # type: ignore
            chat_template=self.custom_template,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        if self.config.train_on_inputs:
            return tokenized_out
        else:
            labels, assistant_mask = (
                tokenized_out["input_ids"],
                tokenized_out["assistant_masks"],
            )
            labels, assistant_mask = torch.tensor(labels), torch.tensor(assistant_mask)
            labels[assistant_mask == 0] = MASK_IDX
            return {
                "input_ids": tokenized_out["input_ids"],
                "attention_mask": tokenized_out["attention_mask"],
                "labels": labels,
            }

    def visualize_sample(self, input_dict) -> Text:
        input_ids = input_dict["input_ids"].squeeze().tolist()
        labels = input_dict["labels"].squeeze().tolist()

        rich_text = Text()

        for token, label in zip(input_ids, labels):
            decoded = self.tokenizer.decode(token)
            if label == 0 or label == MASK_IDX:
                rich_text.append(decoded, style="bright_red")
            else:
                rich_text.append(decoded, style="bright_green")
        return rich_text


class ConversationRawDataModule(ConversationDataModule):
    """
    Like the ConversationDataModule, but does not tokenize the conversation.
    """

    def process_samples_batch_sft(self, examples: dict):  # type: ignore
        out = self._tokenize_conversation(examples["conversation"])
        return out
