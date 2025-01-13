import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union
import re
import os

import lightning.pytorch as pl
from datasets import Dataset, load_dataset, DatasetDict
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from transformers.tokenization_utils import PreTrainedTokenizer

PROMPT_EXPANSION_TASK_PREFIX = "Expand the following prompt to add more detail: "
SAFETY_TASK_PREFIX = (
    "Rewrite the following prompt to remove any unsafe or copyrighted content: "
)
IGNORE_TOKEN_INDEX = -100
PAD_TOKEN_ID = 0

OptimizerChoice = Literal["AdamW", "Adafactor", "AdamW8bit"]
DataModuleChoice = Literal[
    "ultra_feedback", "code_contests", "conversation", "conversation_dpo"
]
TuningModeChoice = Literal["dpo_lora", "dpo_full", "sft_lora", "sft"]


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
    tuning_type: TuningModeChoice = "sft"

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


def class_name_to_underscore(cls):
    class_name = cls.__name__  # Get the class name
    underscore_case = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    return underscore_case


@dataclass
class DatasetConfig:
    batch_size: int
    max_sequence_length: int
    tuning_mode: TuningModeChoice
    using_mistral: bool
    notebook_mode: bool
    input_dataset_name: Optional[str] = None
    max_samples: Optional[int] = None
    custom_chat_template: Optional[str] = None
    train_on_inputs: bool = False


class SmDataset(pl.LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
        super().__init__()

        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.tokenizer = tokenizer
        self.num_workers = 1 if config.notebook_mode else 4
        current_dir = Path().resolve().name
        self.prefix = ""
        if current_dir == "notebooks":
            self.prefix = "../"
        self.cache_dir = (
            f"{self.prefix}dataset_caches/{class_name_to_underscore(self.__class__)}"
        )
        logger.info(f"Cache dir: {self.cache_dir}")
        self.input_column, self.target_column = "context", "fields"
        self.train_dataset = None
        self.val_dataset = None
        self.dataset_name = None

    def init_dataset(self):
        """
        Cannot call load_dataset as that will shadow the load_dataset function from datasets
        """
        # Load dataset and split
        assert (
            self.config.input_dataset_name is not None
        ), "Input dataset name must be set"
        local_dataset_location = f"{self.prefix}{self.config.input_dataset_name}"
        if os.path.exists(local_dataset_location):
            logger.info(f"Loading local dataset from {local_dataset_location}")
            dataset = Dataset.from_parquet(local_dataset_location)
        else:
            logger.info(f"Dataset not found at expected local location {local_dataset_location}, loading from remote: {self.config.input_dataset_name}")
            dataset = load_dataset(self.config.input_dataset_name)
            if isinstance(dataset, DatasetDict):
                dataset = dataset["train"]
        dataset = dataset.train_test_split(test_size=0.1)  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.custom_template = None
        if self.config.custom_chat_template is not None:
            chat_template_path = (
                f"{self.prefix}chat_templates/{self.config.custom_chat_template}.jinja"
            )
            logger.info(f"Loading custom chat template: {chat_template_path}")
            with open(chat_template_path) as f:
                self.custom_template = f.read()

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")
        ensure_directory(self.cache_dir, clear=False)
        use_cache = not self.config.notebook_mode
        logger.info(
            f"Processing dataset for stage {stage}, workers: {self.num_workers}, cache dir {self.cache_dir}, using cache: {use_cache}"
        )

        self.init_dataset()

        assert self.train_dataset is not None
        assert self.val_dataset is not None
        logger.info(
            f"Train dataset size: {len(self.train_dataset)} Val dataset size: {len(self.val_dataset)}"
        )

        self.train_dataset = self.train_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=use_cache,
            cache_file_name=f"{self.cache_dir}/training.parquet",
            num_proc=self.num_workers,
        )

        self.val_dataset = self.val_dataset.map(
            self.process_samples_batch,
            batched=True,
            load_from_cache_file=use_cache,
            cache_file_name=f"{self.cache_dir}/validation.parquet",
            num_proc=self.num_workers,
        )

    def post_setup(self):
        pass

    def process_samples_batch(self, examples: dict):
        return self._tokenize(examples[self.input_column], examples[self.target_column])

    def _tokenize(self, inputs: List[str], labels: List[str]) -> dict:
        inputs_tokenized = self.tokenizer(
            inputs,
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels_tokenized = self.tokenizer(
            labels,
            max_length=self.config.max_sequence_length,
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
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
        )  # type: ignore

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


def short_hash(input_string: str, truncate_to: int = 8) -> str:
    hash_object = hashlib.sha256(input_string.encode())
    full_hash = hash_object.hexdigest()
    short_hash = full_hash[:truncate_to]
    return short_hash


def save_dataclass_to_json(dataclass_instance, file_path: str):
    with open(file_path, "w") as file:
        json.dump(asdict(dataclass_instance), file, indent=4)
