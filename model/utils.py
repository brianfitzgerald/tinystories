from typing import List, Literal, Optional, Union
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from pathlib import Path
import shutil
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
from dataclasses import dataclass
from enum import Enum

from transformers.tokenization_utils import PreTrainedTokenizer

PROMPT_EXPANSION_TASK_PREFIX = "Expand the following prompt to add more detail: "
SAFETY_TASK_PREFIX = (
    "Rewrite the following prompt to remove any unsafe or copyrighted content: "
)
IGNORE_TOKEN_INDEX = -100
PAD_TOKEN_ID = 0

OptimizerChoice = Literal["AdamW", "Adafactor", "AdamW8bit"]


class ModelChoice(Enum):
    T5 = "t5"
    LLAMA = "llama"
    SIMPLE_BERT = "simple_bert"
    GPT = "gpt"


@dataclass
class LanguageModelHyperParams:
    base_model_checkpoint: str = "google/flan-t5-small"
    tokenizer_checkpoint: Optional[str] = "google/flan-t5-small"
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
    max_grad_norm: float = 1.0
    seed: int = 42
    weight_decay: float = 0.0
    optimizer: OptimizerChoice = "AdamW8bit"

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
    ):
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.cpu_count = max(len(os.sched_getaffinity(0)), 32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, num_workers=self.cpu_count)  # type: ignore


class SmModel(pl.LightningModule):
    def __init__(self, hparams: LanguageModelHyperParams, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.params = hparams
        self.tokenizer = tokenizer


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
