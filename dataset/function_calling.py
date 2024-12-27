from datasets import load_dataset
from typing import Optional, List, Dict

from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import IGNORE_TOKEN_INDEX, ensure_directory, SmDataset
import os
import re
import torch
from torch import Tensor
import torch.nn.functional as F

from synthetic_data.conversion import chatml_to_conversation


class FunctionCallingDataModule(SmDataset):

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        dataset = load_dataset("glaiveai/glaive-function-calling-v2")["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

        cache_dir = "dataset_caches/function_calling"
        ensure_directory(cache_dir, clear=False)
        cpu_count = min(len(os.sched_getaffinity(0)), 16)  # type: ignore
        # cpu_count = 1

        # Set format for PyTorch
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

        self.train_dataset = self.train_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            num_proc=cpu_count,
            cache_file_name=f"{cache_dir}/training.parquet",
        )

        self.val_dataset = self.val_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            num_proc=cpu_count,
            cache_file_name=f"{cache_dir}/validation.parquet",
        )

    def process_samples_batch(self, examples: dict):
        """
        Parse chatml string to conversation steps, convert to prompt and output, and then tokenize
        Tokenizing is split from applying the chat template so we can output the attention mask
        """

        input_ids, attention_masks, labels = [], [], []
        for system, chat in zip(examples["system"], examples["chat"]):
            conversation_string = system + chat
            conversation_chatml = chatml_to_conversation(conversation_string)

            # N-1 messages are the prompt, then the last message is the expected output
            prompt_str: str = self.tokenizer.apply_chat_template(
                conversation_chatml[:-1], tokenize=False
            )  # type: ignore

            expected_output_str: str = self.tokenizer.apply_chat_template(
                [conversation_chatml[-1]], tokenize=False
            )  # type: ignore

            prompt_tokenized = self.tokenizer(
                prompt_str,
                max_length=self.max_token_length,
                truncation=True,
                return_tensors="pt",
            )

            expected_output_tokenized = self.tokenizer(
                expected_output_str,
                max_length=self.max_token_length,
                truncation=True,
                return_tensors="pt",
            )

            prompt_input_ids: Tensor = prompt_tokenized["input_ids"].squeeze()  # type: ignore
            pad_amt = max(0, self.max_token_length - prompt_input_ids.shape[0])

            attention_mask: Tensor = prompt_tokenized["attention_mask"].squeeze()  # type: ignore
            attention_mask = F.pad(attention_mask, (0, pad_amt), value=0)

            expected_output_input_ids: Tensor = expected_output_tokenized["input_ids"].squeeze()  # type: ignore

            tokenized_prompt_len = prompt_input_ids.shape[0]
            ignore_labels = torch.full((tokenized_prompt_len,), IGNORE_TOKEN_INDEX)
            label = torch.cat([ignore_labels, expected_output_input_ids], dim=0)
            label_pad_amt = max(0, self.max_token_length - label.shape[0])
            label = F.pad(label, (0, label_pad_amt), value=0)
            label = label[: self.max_token_length]

            # pad after concatting to the labels
            prompt_input_ids = F.pad(prompt_input_ids, (0, pad_amt), value=0)

            input_ids.append(prompt_input_ids)
            attention_masks.append(attention_mask)
            labels.append(label)  # type: ignore

        labels_t = torch.stack(labels)
        input_ids_t = torch.stack(input_ids)
        attention_masks_t = torch.stack(attention_masks)
        return {
            "input_ids": input_ids_t,
            "attention_mask": attention_masks_t,
            "labels": labels_t,
        }
