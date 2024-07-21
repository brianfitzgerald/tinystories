from loguru import logger

logger.level("INFO")
logger.info("Loading dependencies - Torch...")
import random
import string
from dataclasses import dataclass

from dotenv import load_dotenv
from fire import Fire
from lightning import seed_everything
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

logger.info("Loading dependencies - Lightning...")
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger

logger.info("Loading dependencies - Project...")
from dataset.function_calling import FunctionCallingDataModule
from dataset.squad import SquadExtractiveQADataModule, SquadDataModule
from dataset.parti import PromptUpsampleDataModule
from dataset.pretrain import BertPretrainDataset, TinyStoriesDataset
from model.callbacks import (
    GradientNormLogger,
    HfModelCheckpoint,
    LogLLMPredictionSamplesCallback,
)
from model.llama import LlamaFineTuner
from model.pretrain.bert import SimpleBertForMaskedLM, get_sane_normalizers
from model.pretrain.gpt import GPT
from model.t5 import T5FineTuner
from model.utils import LanguageModelHyperParams, ModelChoice, SmDataset, SmModel

MODEL_CHOICES = {
    SimpleBertForMaskedLM: ModelChoice.SIMPLE_BERT,
    T5FineTuner: ModelChoice.T5,
    LlamaFineTuner: ModelChoice.LLAMA,
    GPT: ModelChoice.GPT,
}


@dataclass
class ModelConfig:
    model: type[SmModel]
    data_module: type[SmDataset]
    wandb_project_name: str
    hyperparams: LanguageModelHyperParams


PROMPT_UPSAMPLING_PROJECT = "t5-prompt-upsampling"
PROMPT_SAFETY_PROJECT = "t5-prompt-safety"
EXTRACTIVE_QA_PROJECT = "t5-extractive-qa"
SQUAD_QA_PROJECT = "t5-squad-qa"

CONFIGS = {
    "fn_calling": ModelConfig(
        LlamaFineTuner,
        FunctionCallingDataModule,
        "llama-function-calling",
        LanguageModelHyperParams("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ),
    "prompt_upsample_small": ModelConfig(
        T5FineTuner,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        LanguageModelHyperParams(base_model_checkpoint="google/flan-t5-small"),
    ),
    "prompt_upsample": ModelConfig(
        T5FineTuner,
        PromptUpsampleDataModule,
        PROMPT_UPSAMPLING_PROJECT,
        LanguageModelHyperParams(base_model_checkpoint="google/flan-t5-base"),
    ),
    "simple_bert_pretrain": ModelConfig(
        SimpleBertForMaskedLM,
        BertPretrainDataset,
        "simple-bert-pretrain",
        LanguageModelHyperParams(
            # base model is only used for tokenizer
            base_model_checkpoint="bert-base-uncased",
            learning_rate=1e-3,
            warmup_ratio=0.5,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=1,
            train_batch_size=128,
            gradient_accumulation_steps=16,
            max_seq_length=512,
        ),
    ),
    "tiny_stories": ModelConfig(
        GPT,
        TinyStoriesDataset,
        "tinystories-gpt-pretrain",
        LanguageModelHyperParams(
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=1,
            train_batch_size=32,
            val_batch_size=1,
            gradient_accumulation_steps=16,
            max_seq_length=512,
            tokenizer_checkpoint="EleutherAI/gpt-neox-20b",
        ),
    ),
    "info_extraction": ModelConfig(
        T5FineTuner,
        SquadExtractiveQADataModule,
        EXTRACTIVE_QA_PROJECT,
        LanguageModelHyperParams(
            base_model_checkpoint="google/flan-t5-base",
            warmup_ratio=0.2,
            optimizer="Adafactor",
        ),
    ),
    "squad_t5": ModelConfig(
        T5FineTuner,
        SquadDataModule,
        SQUAD_QA_PROJECT,
        LanguageModelHyperParams(
            base_model_checkpoint="google/flan-t5-base",
            learning_rate=3e-4,
            warmup_ratio=0.2,
            optimizer="Adafactor",
            train_batch_size=32,
            val_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=100,
            max_seq_length=512,
        ),
    ),
}


def main(wandb: bool = False, config: str = "squad_t5"):

    load_dotenv(".env")

    loggers = []

    model_config = CONFIGS[config]
    hparams = model_config.hyperparams
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore
        hparams.tokenizer_checkpoint_value
    )
    data_module = model_config.data_module(
        hparams.train_batch_size, tokenizer, hparams.max_seq_length
    )
    model = model_config.model(hparams, tokenizer)

    wandb_logger = None
    run_name = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    run_name = f"{config}-{run_name}"

    if wandb:
        wandb_logger = WandbLogger(
            name=run_name, project=model_config.wandb_project_name
        )
        loggers.append(wandb_logger)
        wandb_logger.watch(model)
    else:
        loggers.append(CSVLogger("logs", name=run_name))

    model_choice: ModelChoice = MODEL_CHOICES[model.__class__]  # type: ignore
    sample_callback = LogLLMPredictionSamplesCallback(
        tokenizer, model_choice, wandb_logger
    )

    learning_rate_callback = LearningRateMonitor(logging_interval="step")
    gradient_norm_callback = GradientNormLogger()

    if model_choice == ModelChoice.SIMPLE_BERT:
        tokenizer._tokenizer.normalizer = get_sane_normalizers(  # type: ignore
            force_english_keyboard=True,
            strip_accents=True,
            force_lowercase=True,
        )

    if model_choice == ModelChoice.GPT:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    seed_everything(hparams.seed)

    if model_choice in (ModelChoice.SIMPLE_BERT, ModelChoice.GPT):
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=run_name,
            monitor="val_loss",
            mode="min",
        )
    else:
        checkpoint_callback = HfModelCheckpoint(
            dirpath="checkpoints",
            filename=run_name,
            monitor="val_loss",
            mode="min",
        )

    progress_bar_callback = TQDMProgressBar(refresh_rate=1)
    precision = "32" if model_config.model == T5FineTuner else "16-mixed"

    effective_batch_size = (
        hparams.train_batch_size * hparams.gradient_accumulation_steps
    )
    logger.info(
        f"Effective batch size: {effective_batch_size} ({hparams.gradient_accumulation_steps}x{hparams.train_batch_size})"
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision=precision,
        gradient_clip_val=hparams.max_grad_norm,
        val_check_interval=0.1,
        callbacks=[
            sample_callback,
            checkpoint_callback,
            progress_bar_callback,
            learning_rate_callback,
            gradient_norm_callback,
        ],
        logger=loggers,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(main)
