import torch
from rich.text import Text

from model.utils import (
    SmDataset,
)
from synthetic_data.utils import Conversation

MASK_IDX = -100


class ConversationDataModule(SmDataset):
    """
    Generic data module for conversation datasets.
    """

    def post_setup(self):
        self.train_dataset = self.train_dataset.remove_columns("conversation")
        self.val_dataset = self.val_dataset.remove_columns("conversation")

    def process_samples_batch(self, examples: dict):
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


class ConversationDPODataModule(ConversationDataModule):
    """
    Conversation data module, assumes a dataset with `chosen`, `rejected`, and `prompt` columns.
    The columns are formatted into a conversation and tokenized.
    """

    def process_samples_batch(self, examples):
        batch_out = {
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
            "prompt": examples["prompt"],
        }
        return batch_out
