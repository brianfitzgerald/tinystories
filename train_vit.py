import random
import string
from dataclasses import dataclass
from typing import Optional

import lightning.pytorch as pl
from fire import Fire
from lightning import seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger

from dataset.aesthetic_score import AestheticScoreDataset, Cifar10Dataset, VitDataset
from model.callbacks import GradientNormLogger
from model.vit import VisionTransformer, VitHParams


@dataclass
class VitModelConfig:
    wandb_project_name: str
    data_module: type[VitDataset]
    hyperparams: VitHParams


CONFIGS = {
    "vit": VitModelConfig(
        "aesthetic-scorer-vit",
        AestheticScoreDataset,
        VitHParams(
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=10,
            train_batch_size=32,
            val_batch_size=16,
            gradient_accumulation_steps=16,
        ),
    ),
    "cifar10": VitModelConfig(
        "cifar10-vit",
        Cifar10Dataset,
        VitHParams(
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=0.5,
            num_train_epochs=10,
            train_batch_size=32,
            val_batch_size=16,
            gradient_accumulation_steps=16,
            depth=6,
            n_heads=8,
            mlp_dim=512,
            dropout=0.1
        ),
    )
}


def main(wandb: bool = False, config: str = "cifar10"):
    loggers = []

    model_config = CONFIGS[config]
    hparams = model_config.hyperparams
    data_module = model_config.data_module(model_config.hyperparams.train_batch_size)
    model = VisionTransformer(hparams, data_module)

    wandb_logger: Optional[WandbLogger] = None
    run_name = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    run_name = f"{config}-{run_name}"
    logger.info(f"Starting run {run_name}")

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

    progress_bar_callback = TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.gradient_accumulation_steps,
        max_epochs=hparams.num_train_epochs,
        precision="16-mixed",
        gradient_clip_val=hparams.max_grad_norm,
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
