from model.utils import ensure_directory, SmDataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
from typing import Optional


class VitDataset(pl.LightningDataModule):
    image_size: int = 224
    patch_size: int = 16
    n_classes: int = 1000


class AestheticScoreDataset(VitDataset):
    def __init__(
        self,
        batch_size: int,
    ):
        self.batch_size = batch_size
        self.cpu_count = max(len(os.sched_getaffinity(0)), 32)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cpu_count)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8, num_workers=self.cpu_count)  # type: ignore

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/ava"

        ensure_directory(cache_dir, clear=False)

        assert self.train_dataset is not None
        assert self.val_dataset is not None

        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/training.parquet",
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            batched=True,
            load_from_cache_file=True,
            num_proc=self.cpu_count,
            cache_file_name=f"{cache_dir}/validation.parquet",
        )

    def prepare_sample(self, examples: dict):

        return {}
