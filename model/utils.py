from typing import List, Literal, Optional, Union, Dict
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from pathlib import Path
import shutil
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
from dataclasses import dataclass
from enum import Enum
from loguru import logger
from datasets import load_dataset

from transformers.tokenization_utils import PreTrainedTokenizer

PROMPT_EXPANSION_TASK_PREFIX = "Expand the following prompt to add more detail: "
SAFETY_TASK_PREFIX = (
    "Rewrite the following prompt to remove any unsafe or copyrighted content: "
)
IGNORE_TOKEN_INDEX = -100
PAD_TOKEN_ID = 0

OptimizerChoice = Literal["AdamW", "Adafactor", "AdamW8bit"]
TuningType = Literal["sft", "dpo"]


class ModelChoice(Enum):
    T5 = "t5"
    CAUSAL_LM = "causal_lm"
    SIMPLE_BERT = "simple_bert"
    GPT = "gpt"


@dataclass
class LMHyperParams:
    base_model_checkpoint: str = "google/flan-t5-small"
    tokenizer_checkpoint: Optional[str] = None
    max_seq_length: int = 2048
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    warmup_steps_count: Optional[int] = None
    warmup_ratio: Optional[float] = None
    train_batch_size: int = 4
    val_batch_size: int = 2
    num_train_epochs: int = 25
    gradient_accumulation_steps: int = 2
    n_gpus: int = 1
    max_grad_norm: Optional[float] = None
    seed: int = 42
    weight_decay: float = 0.0
    optimizer: OptimizerChoice = "AdamW8bit"
    tuning_type: TuningType = "sft"

    def warmup_steps(self, train_steps: Union[int, float]) -> int:
        if self.warmup_ratio:
            return int(self.warmup_ratio * train_steps)
        elif self.warmup_steps_count:
            return self.warmup_steps_count
        else:
            raise ValueError("Either warmup_steps_count or warmup_ratio must be set")

    @property
    def tokenizer_checkpoint_value(self) -> str:
        if self.tokenizer_checkpoint:
            return self.tokenizer_checkpoint
        return self.base_model_checkpoint


class SmDataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        max_token_length: int,
        use_cache: bool = True,
    ):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.num_workers = min(len(os.sched_getaffinity(0)), 8)
        self.num_workers = 1
        self.cache_dir = "dataset_caches/default"
        self.input_column, self.target_column = "context", "fields"
        self.use_cache = use_cache
        self.dataset_name = "roborovski/squad-extractive-qa"
        self.train_dataset = None
        self.val_dataset = None

    def load_dataset(self):
        # Load dataset and split
        dataset = load_dataset(self.dataset_name)["train"].train_test_split(test_size=0.01)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")
        ensure_directory(self.cache_dir, clear=False)
        logger.info(
            f"Processing dataset for stage {stage}, workers: {self.num_workers}, cache dir {self.cache_dir}"
        )

        self.load_dataset()

        assert self.train_dataset is not None
        assert self.val_dataset is not None

        self.train_dataset = self.train_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=self.use_cache,
            cache_file_name=f"{self.cache_dir}/training.parquet",
            num_proc=self.num_workers,
        )

        self.val_dataset = self.val_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=self.use_cache,
            cache_file_name=f"{self.cache_dir}/validation.parquet",
            num_proc=self.num_workers,
        )

        if all(
            [
                x in self.train_dataset.column_names
                for x in ["input_ids", "attention_mask", "labels"]
            ]
        ):
            columns = [
                "input_ids",
                "attention_mask",
                "labels",
            ]

            # Set format for PyTorch
            self.train_dataset.set_format(type="torch", columns=columns)
            self.val_dataset.set_format(type="torch", columns=columns)
        else:
            logger.warning(
                f"Computed columns not found in dataset: {self.train_dataset.column_names} assuming using "
            )

    def process_samples_batch(self, examples: dict):
        return self._tokenize(examples[self.input_column], examples[self.target_column])

    def _tokenize(self, inputs: List[str], labels: List[str]) -> dict:
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
            "labels": labels_tokenized["input_ids"],
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, num_workers=self.num_workers)  # type: ignore


class SmModel(pl.LightningModule):
    def __init__(self, hparams: LMHyperParams) -> None:
        super().__init__()
        self.params = hparams
        self.model_choice = ModelChoice.CAUSAL_LM


def compute_metrics(inputs: List[str], generated: List[str]):
    rouge = ROUGEScore()
    bleu = BLEUScore()

    rouge_scores = rouge(inputs, generated)
    bleu_score = bleu(inputs, generated)

    return {
        **rouge_scores,
        "bleu": bleu_score.item(),
    }


def ensure_directory(directory: str, clear: bool = True):
    """
    Create a directory and parents if it doesn't exist, and clear it if it does.
    """
    Path(directory).mkdir(exist_ok=True, parents=True)
    if clear:
        shutil.rmtree(directory)
    Path(directory).mkdir(exist_ok=True, parents=True)
