print("Loading dependencies - torch...")
from fire import Fire
from dataclasses import dataclass

from lightning import seed_everything
import random
import string

from model.t5 import T5FineTuner
from model.pretrain.gpt import GPT
from model.utils import SmModel, VitHyperParams

print("Loading dependencies - lightning...")
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import lightning.pytorch as pl
from model.callbacks import (
    GradientNormLogger,
)
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)


print("Loading dependencies - project...")
from model.utils import ModelChoice, SmDataset, VitHyperParams


MODEL_CHOICES = {
    GPT: ModelChoice.GPT,
}


@dataclass
class VitModelConfig:
    wandb_project_name: str
    hyperparams: VitHyperParams = VitHyperParams()


PROMPT_UPSAMPLING_PROJECT = "t5-prompt-upsampling"
PROMPT_SAFETY_PROJECT = "t5-prompt-safety"

CONFIGS = {
    "vit": VitModelConfig(
        "aesthetic-scorer-vit",
        VitHyperParams(
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=1,
            train_batch_size=32,
            val_batch_size=1,
            gradient_accumulation_steps=16,
        ),
    ),
}


def main(wandb: bool = False, config: str = "vit"):
    loggers = []

    model_config = CONFIGS[config]
    hparams = model_config.hyperparams
    data_module = model_config.data_module(
        hparams.train_batch_size
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

    learning_rate_callback = LearningRateMonitor(logging_interval="step")
    gradient_norm_callback = GradientNormLogger()

    seed_everything(hparams.seed)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=run_name,
        monitor="val_loss",
        mode="min",
    )

    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    precision = "32" if model_config.model == T5FineTuner else "16-mixed"

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision=precision,
        gradient_clip_val=hparams.max_grad_norm,
        val_check_interval=0.1,
        callbacks=[
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
