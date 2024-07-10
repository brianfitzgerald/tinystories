from model.utils import ensure_directory
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
from typing import Optional
from datasets import load_dataset, DatasetDict
from torchvision.transforms import transforms
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import resize, center_crop, to_tensor
import torch


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
        self.proc_count = max(len(os.sched_getaffinity(0)), 32)
        logger.info(f"Using {self.proc_count} processes for data loading")
        dataset: DatasetDict = load_dataset("THUDM/ImageRewardDB", "1k")  # type: ignore
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=8,  collate_fn=self.collate_fn)  # type: ignore

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/image_reward"

        ensure_directory(cache_dir, clear=False)

        assert self.train_dataset is not None
        assert self.val_dataset is not None

        logger.info("Filtering invalid images...")

    def filter_valid_images(self, batch: dict):
        if "image" not in batch:
            return False
        try:
            image = batch["image"]
            image_pil = Image.fromarray(image)
            if image_pil.mode != "RGB":
                return False
            if image_pil.format not in ["JPEG", "PNG"]:
                return False
        except:
            return False
        return True

    def collate_fn(self, batch):
        image_tensors = []
        labels = []
        for sample in batch:
            image = to_tensor(sample["image"])
            image = resize(image, [self.image_size])
            image = center_crop(image, [self.image_size, self.image_size])
            image = image.to(dtype=torch.float32)
            image_tensors.append(image)

            labels.append(sample["overall_rating"])
        image_tensors = torch.stack(image_tensors)
        labels = torch.tensor(labels)
        return {"image": image_tensors, "label": labels}

    def prepare_sample(self, batch: dict):
        image = batch["image"]
        image = self.transforms(image)

        out = {
            "image": image,
        }

        for key in self.COLUMNS:
            out[key] = batch[key]
        return out
