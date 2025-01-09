from typing import List, Optional

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
from synthetic_data.utils import Conversation, ldictl
import os


class ConversationDataModule(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__(tokenizer, config)

        current_dir = Path().resolve().name
        prefix = ""
        if current_dir == "notebooks":
            prefix = "../"
        assert self.config.input_dataset_name is not None
        input_dataset_name = self.config.input_dataset_name
        model_name = self.tokenizer.name_or_path  # type: ignore
        if os.path.isfile(self.config.input_dataset_name):
            input_dataset_name = os.path.basename(self.config.input_dataset_name)
        self.cache_dir = f"{prefix}dataset_caches/{class_name_to_underscore(self.__class__)}/{model_name}/{input_dataset_name}"
        logger.info(f"Cache dir: {self.cache_dir}")
        self.train_on_inputs = config.train_on_inputs

    def load_dataset(self):
        # Load dataset and split
        logger.info("Loading dataset")
        assert self.config.input_dataset_name is not None
        dataset = Dataset.from_parquet(self.config.input_dataset_name)
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.custom_template = None
        if self.config.custom_chat_template is not None:
            with open(f"chat_templates/{self.config.custom_chat_template}.jinja") as f:
                self.custom_template = f.read()

    def post_setup(self):
        self.train_dataset = self.train_dataset.remove_columns("conversation")
        self.val_dataset = self.val_dataset.remove_columns("conversation")

    def process_samples_batch_sft(self, examples: dict):
        out = self.tokenize_conversation(examples["conversation"])
        return out

    def _tokenize_conversation(self, conversation: Conversation) -> List[int]:
        return self.tokenizer.apply_chat_template(
            conversation,  # type: ignore
            chat_template=self.custom_template,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )

    def tokenize_conversation(self, conversations: List[Conversation]):
        tokenized_batch = []
        for all_turns in conversations:
            all_but_last_turn = all_turns[:-1]
            prompt_ids = self._tokenize_conversation(all_but_last_turn)
            all_turn_ids: List[int] = self._tokenize_conversation(all_turns)
            attn_mask = [1] * len(prompt_ids)
            tokenized_prompt = {}
            input_ids = []

            if self.train_on_inputs:
                labels = all_turn_ids
                input_ids = all_turn_ids
            else:
                user_prompt_len = len(prompt_ids)
                input_ids: List[int] = prompt_ids + all_turn_ids[user_prompt_len:]  # type: ignore
                labels: List[int] = [-100] * user_prompt_len + all_turn_ids[
                    user_prompt_len:
                ]  # type: ignore
                if len(prompt_ids) < self.config.max_sequence_length:
                    # logger.warning(
                    #     f"Prompt length {len(prompt_ids)} is less than max_sequence_length {self.config.max_sequence_length}, padding"
                    # )
                    input_ids = input_ids + [0] * (
                        self.config.max_sequence_length - len(input_ids)
                    )
                    labels = labels + [-100] * (
                        self.config.max_sequence_length - len(labels)
                    )
                    attn_mask = attn_mask + [0] * (
                        self.config.max_sequence_length - len(attn_mask)
                    )

            tokenized_prompt["labels"] = labels
            tokenized_prompt["input_ids"] = input_ids
            tokenized_prompt["attention_mask"] = attn_mask

            tokenized_batch.append(tokenized_prompt)
            # print([(len(v), k) for k, v in tokenized_prompt.items()])

        batch = ldictl(tokenized_batch)
        return batch

    def visualize_sample(self, input_dict) -> Text:
        input_ids = input_dict["input_ids"].squeeze().tolist()
        labels = input_dict["labels"].squeeze().tolist()

        rich_text = Text()

        for token, label in zip(input_ids, labels):
            decoded = self.tokenizer.decode(token)
            if label == 0 or label == -100:
                rich_text.append(decoded, style="bright_red")
            else:
                rich_text.append(decoded, style="bright_green")
        return rich_text


class ConversationRawDataModule(ConversationDataModule):
    """
    Like the ConversationDataModule, but does not tokenize the conversation.
    """
    def process_samples_batch_sft(self, examples: dict): # type: ignore
        out = self._tokenize_conversation(examples["conversation"])
        return out