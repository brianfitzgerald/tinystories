from datasets import load_dataset, concatenate_datasets, Dataset, ReadInstruction
from typing import Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import ensure_directory, SmDataset
import re
from unidecode import unidecode
import torch
from torch import Tensor


def clean_bookcorpus_text(text: str) -> str:
    s = unidecode(text)
    s = s.lower()
    s = re.sub(
        "[ \t]+", " ", s
    )  # Replace tabs and sequences of spaces with a single space
    s = s.replace("\n", "\\n")
    return s.strip()


class BertPretrainDataset(SmDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.max_token_length = max_token_length
        self.mlm_probability = 0.15
        self.pad_token_id: int = self.tokenizer.pad_token_id  # type: ignore
        self.mask_token_id: int = self.tokenizer.mask_token_id  # type: ignore

    def prepare_data(self) -> None:
        # bc: Dataset = load_dataset("saibo/bookcorpus_deduplicated_small", split="train")  # type: ignore
        TEST_SPLIT = False
        split = (
            ReadInstruction("train", to=1000)
            if TEST_SPLIT
            else ReadInstruction("train")
        )
        bc: Dataset = load_dataset("sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen", split=split)  # type: ignore
        # wp: Dataset = load_dataset("wikipedia", "20220301.en", split="train[0:100000]")  # type: ignore

        self.full_dataset = concatenate_datasets([bc]).train_test_split(test_size=0.01)

        self.train_dataset = self.full_dataset["train"]
        self.val_dataset = self.full_dataset["test"]
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/bert_pretrain"

        ensure_directory(cache_dir, clear=False)

        assert self.train_dataset is not None
        assert self.val_dataset is not None

        self.train_dataset = self.train_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/training.parquet",
        )

        self.val_dataset = self.val_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/validation.parquet",
        )

    def mask_tokens(self, inputs: Tensor, special_tokens_mask) -> Tuple[Tensor, Tensor]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = self.mask_token_id

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(  # type: ignore
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def process_samples_batch(self, examples: dict):
        input_ids = [clean_bookcorpus_text(doc) for doc in examples["text"]]

        inputs_tokenized = self.tokenizer(
            input_ids,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        input_ids, labels = self.mask_tokens(
            inputs_tokenized["input_ids"],  # type: ignore
            inputs_tokenized["special_tokens_mask"],
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": inputs_tokenized["attention_mask"],
        }


class TinyStoriesDataset(SmDataset):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
    ):
        super().__init__(batch_size, tokenizer, max_token_length)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.max_token_length = max_token_length
        # self.cpu_count = 1

    def prepare_data(self) -> None:
        TEST_SPLIT = False
        split = (
            ReadInstruction("train", to=1000)
            if TEST_SPLIT
            else ReadInstruction("train")
        )
        bc: Dataset = load_dataset("roneneldan/TinyStories", split=split)  # type: ignore

        self.full_dataset = concatenate_datasets([bc]).train_test_split(test_size=0.01)

        self.train_dataset = self.full_dataset["train"]
        self.val_dataset = self.full_dataset["test"]
        self.train_dataset.set_format(type="torch")
        self.val_dataset.set_format(type="torch")

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/tinystories_pretrain"

        ensure_directory(cache_dir, clear=False)

        assert self.train_dataset is not None
        assert self.val_dataset is not None

        self.train_dataset = self.train_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/training.parquet",
        )

        self.val_dataset = self.val_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/validation.parquet",
        )

    def process_samples_batch(self, examples: dict):
        input_text = [clean_bookcorpus_text(doc) for doc in examples["text"]]

        inputs_tokenized = self.tokenizer(
            input_text,
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs_tokenized["input_ids"],
            "labels": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
        }
