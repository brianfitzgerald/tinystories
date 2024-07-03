from model.utils import ensure_directory
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
from typing import Optional
from datasets import load_dataset, DatasetDict
from torchvision.transforms import transforms
from PIL import Image


class VitDataset(pl.LightningDataModule):
    image_size: int = 224
    patch_size: int = 16
    n_classes: int = 1000

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        super().__init__()


class AestheticScoreDataset(VitDataset):
    
    COLUMNS = [
        "image_text_alignment_rating",
        "fidelity_rating",
        "overall_rating",
        "rank",
    ]

    def __init__(
        self,
        batch_size: int,
    ):
        super().__init__(batch_size)
        # self.proc_count = max(len(os.sched_getaffinity(0)), 32)
        self.proc_count = 8
        dataset: DatasetDict = load_dataset("THUDM/ImageRewardDB", "1k")  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
            ]
        )
        self.train_dataset.set_format(type="torch", columns=self.COLUMNS + ["image"])
        self.val_dataset.set_format(type="torch", columns=self.COLUMNS + ["image"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8)  # type: ignore

    def setup(self, stage: Optional[str] = None):
        print(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/image_reward"

        ensure_directory(cache_dir, clear=False)

        assert self.train_dataset is not None
        assert self.val_dataset is not None

        self.train_dataset = self.train_dataset.map(
            self.prepare_sample,
            cache_file_name=os.path.join(cache_dir, "train.parquet"),
            num_proc=self.proc_count,
            load_from_cache_file=True,
        )

        self.val_dataset = self.val_dataset.map(
            self.prepare_sample,
            cache_file_name=os.path.join(cache_dir, "val.parquet"),
            num_proc=self.proc_count,
            load_from_cache_file=True,
        )

    def prepare_sample(self, batch: dict):
        try:
            image = batch["image"]
        except:
            image = Image.new("RGB", (self.image_size, self.image_size))

        image = self.transforms(image)

        out = {
            "image": image,
        }

        for key in self.COLUMNS:
            out[key] = batch[key]
        return out
