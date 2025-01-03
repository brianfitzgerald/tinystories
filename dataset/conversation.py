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
        custom_template = None
        if self.config.custom_chat_template is not None:
            with open(
                f"../chat_templates/{self.config.custom_chat_template}.jinja"
            ) as f:
                custom_template = f.read()
        return self.tokenizer.apply_chat_template(
            conversation,  # type: ignore
            chat_template=custom_template,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
        )

    def tokenize_conversation(self, conversations: List[Conversation]):
        tokenized_batch = []
        for all_turns in conversations:
            all_but_last_turn = all_turns[:-1]
            prompt_ids = self._tokenize_conversation(all_but_last_turn)
            all_turn_ids = self._tokenize_conversation(all_turns)
            attn_mask = [1] * len(prompt_ids)
            tokenized_prompt = {}

            if not self.train_on_inputs:
                user_prompt_len = len(prompt_ids)
                input_ids = prompt_ids + all_turn_ids[user_prompt_len:]
                labels = [-100] * user_prompt_len + all_turn_ids[user_prompt_len:]
                if len(prompt_ids) < self.config.max_sequence_length:
                    logger.warning(
                        f"Prompt length {len(prompt_ids)} is less than max_sequence_length {self.config.max_sequence_length}, padding"
                    )
                    input_ids = input_ids + [0] * (
                        self.config.max_sequence_length - len(input_ids)
                    )
                    labels = labels + [-100] * (
                        self.config.max_sequence_length - len(labels)
                    )
                    attn_mask = attn_mask + [0] * (
                        self.config.max_sequence_length - len(attn_mask)
                    )
            else:
                labels = input_ids

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
                rich_text.append(decoded, style="bright_green")
            else:
                rich_text.append(decoded, style="bright_red")
        return rich_text
