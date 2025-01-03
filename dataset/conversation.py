from typing import List

from datasets.arrow_dataset import Dataset
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer
from rich.text import Text

from model.utils import DatasetConfig, SmDataset
from synthetic_data.utils import Conversation, ldictl


class ConversationDataModule(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__(tokenizer, config)
        self.train_on_inputs = False

    def load_dataset(self):
        # Load dataset and split
        logger.info("Loading dataset")
        assert self.config.input_dataset_name is not None
        dataset = Dataset.from_parquet(self.config.input_dataset_name)
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.messages_field = "conversation"

    def post_setup(self):
        self.train_dataset = self.train_dataset.remove_columns("conversation")
        self.val_dataset = self.val_dataset.remove_columns("conversation")

    def process_samples_batch_sft(self, examples: dict):
        out = self.tokenize_conversation(examples["conversation"])
        return out

    def _tokenize_conversation(self, conversation: Conversation):
        return self.tokenizer.apply_chat_template(
            conversation,  # type: ignore
            padding=True,
        )

    def tokenize_conversation(self, conversations: List[Conversation]):
        tokenized_batch = []
        for all_turns in conversations:
            all_but_last_turn = all_turns[:-1]
            prompt_ids = self._tokenize_conversation(all_but_last_turn)
            all_turn_ids = self._tokenize_conversation(all_turns)
            tokenized_prompt = {}
            tokenized_prompt["input_ids"] = prompt_ids
            tokenized_prompt["attention_mask"] = [1] * len(prompt_ids)

            if not self.train_on_inputs:
                user_prompt_len = len(prompt_ids)
                input_ids = prompt_ids + all_turn_ids[user_prompt_len:]  # type: ignore
                labels = [-100] * user_prompt_len + all_turn_ids[user_prompt_len:]  # type: ignore
            else:
                input_ids = all_turn_ids
                labels = prompt_ids

            tokenized_prompt["labels"] = labels
            tokenized_prompt["input_ids"] = input_ids

            tokenized_batch.append(tokenized_prompt)

        batch = ldictl(tokenized_batch)
        return batch

    def visualize_sample(self, input_dict) -> Text:
        input_ids = input_dict["input_ids"].squeeze().tolist()
        labels = input_dict["labels"].squeeze().tolist()

        rich_text = Text()

        for token, label in zip(input_ids, labels):
            decoded = self.tokenizer.decode(token)
            if label == 0 or label == -100:
                rich_text.append(decoded, style="bright_green")
            else:
                rich_text.append(decoded, style="bright_red")
        return rich_text
