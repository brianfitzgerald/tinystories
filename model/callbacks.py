from typing import Optional, cast
from tabulate import tabulate
import pandas as pd
from pathlib import Path
import shutil
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from fsspec.core import url_to_fs
from torch import Tensor
import torch

from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
from transformers.tokenization_utils import PreTrainedTokenizer


from model.pretrain.gpt import generate, GPT
from model.utils import (
    IGNORE_TOKEN_INDEX,
    PAD_TOKEN_ID,
    ModelChoice,
)


class LogLLMPredictionSamplesCallback(pl.Callback):
    """
    Log prediction samples when training a language model.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_choice: ModelChoice,
        wandb_logger: Optional[WandbLogger] = None,
        max_new_tokens: int = 256,
    ):
        self.tokenizer = tokenizer
        self.wandb_logger = wandb_logger
        self.max_new_tokens = max_new_tokens
        self.model_choice = model_choice

        # TODO clear existing log files
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        self.log_prediction_samples(trainer, pl_module, outputs, batch, batch_idx, 0)

    def log_prediction_samples(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if batch_idx > 0:
            return
        # TODO implement
        input_ids: Tensor = batch["input_ids"]
        labels: Tensor = batch["labels"]

        n = len(input_ids)
        columns = ["Epoch", "Sample Index", "Input", "Output", "Target"]

        table_columns: list[list] = [[trainer.current_epoch] * n, list(range(n))]
        mask_token_id, pad_token_id = (
            pl_module.tokenizer.mask_token_id,
            pl_module.tokenizer.pad_token_id,
        )

        if self.model_choice == ModelChoice.SIMPLE_BERT:
            logits, _, _ = pl_module(input_ids, labels)
            out = logits[input_ids == mask_token_id].argmax(dim=-1)

            # add input_ids, out, label columns
            table_columns.extend([[], [], []])

            for batch_idx in range(n):
                input_ids_display = input_ids[batch_idx]
                input_ids_display = input_ids_display[input_ids_display != pad_token_id]
                input_ids_decoded = self.tokenizer.decode(input_ids_display)
                table_columns[2].append(input_ids_decoded)

                out_display = out[batch_idx]
                out_display = out_display[out_display != pad_token_id]
                out_display = out_display[out_display != mask_token_id]
                out_decoded = self.tokenizer.decode(out_display)
                table_columns[3].append(out_decoded)

                labels_display = labels[batch_idx]
                labels_display = labels_display[labels_display != pad_token_id]
                labels_display = labels_display[labels_display != mask_token_id]
                labels_decoded = self.tokenizer.decode(labels_display)
                table_columns[4].append(labels_decoded)

        elif self.model_choice == ModelChoice.GPT:
            model = cast(GPT, pl_module)
            model.set_kv_cache(batch_size=1, device=input_ids.device)

            B, T = input_ids.shape
            out_batch = []
            for i in range(B):
                input_ids_sample = input_ids[i, : T - self.max_new_tokens]
                out = generate(model, input_ids_sample, T)
                out_batch.append(out)

            out = torch.stack(out_batch, dim=0)

            for feature in [input_ids, out, labels]:
                decoded = self.tokenizer.batch_decode(
                    feature, clean_up_tokenization_spaces=True
                )
                decoded = [s.replace("[PAD]", "").strip() for s in decoded]
                table_columns.append(decoded)

        else:
            # IGNORE_TOKEN_INDEX is not respected in inference, so replace it with PAD_TOKEN_ID
            labels[labels[:, :] == IGNORE_TOKEN_INDEX] = PAD_TOKEN_ID
            out = pl_module.model.generate(
                input_ids,
                max_length=self.max_new_tokens,
            )

            for feature in [input_ids, out, labels]:
                decoded = self.tokenizer.batch_decode(
                    feature, clean_up_tokenization_spaces=True
                )
                decoded = [s.replace("[PAD]", "").strip() for s in decoded]
                table_columns.append(decoded)

        run_name = "latest"
        if self.wandb_logger:
            run_name = self.wandb_logger.experiment.name
            table_rows = list(zip(*table_columns))
            self.wandb_logger.log_table("Validation Samples", columns, table_rows)
            # metrics = compute_metrics(table_columns[3], table_columns[4])
            # self.wandb_logger.log_metrics(metrics)

        rows = [list(row) for row in zip(*table_columns)]
        rows_df = pd.DataFrame(rows, columns=columns)
        rows_df.to_csv(
            self.log_dir / f"{run_name}_samples.csv",
            mode="a",
            header=False,
            index=False,
        )

        new_rows = tabulate(
            rows,
            headers=columns,
            maxcolwidths=[10, 10, 50, 50, 50],
        )
        print(new_rows)
        with open(self.log_dir / f"{run_name}_samples.txt", "a") as f:
            f.write(new_rows)
            f.write("\n")


# https://github.com/Lightning-AI/pytorch-lightning/issues/3096#issuecomment-1441278197
class HfModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.FILE_EXTENSION = ""

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        print(f"Saving checkpoint at {filepath}")
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(filepath)
            trainer.lightning_module.tokenizer.save_pretrained(filepath)

    # https://github.com/Lightning-AI/lightning/pull/16067
    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        print(f"Removing checkpoint at {filepath}")
        if trainer.is_global_zero:
            fs, _ = url_to_fs(filepath)
            if fs.exists(filepath):
                fs.rm(filepath, recursive=True)


def gradient_norm(model: pl.LightningModule):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


class GradientNormLogger(pl.Callback):
    """
    Logs the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm_total", gradient_norm(model))
