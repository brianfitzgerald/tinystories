import os
from pathlib import Path
from typing import List, Optional
import json
import io

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import webdataset as wds
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.functional import center_crop, resize, to_tensor
from datasets import load_dataset

from model.utils import ensure_directory


def get_wds_file_list(input_dataset: str) -> List[str]:
    if input_dataset.endswith(".tar"):
        all_files_in_dataset = [input_dataset]
    else:
        path = Path(input_dataset)
        all_files_in_dataset: List[str] = []
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".tar":
                all_files_in_dataset.append(str(file_path))
        all_files_in_dataset = [
            os.path.join(input_dataset, file)
            for file in all_files_in_dataset
            if file.endswith(".tar")
        ]
    return all_files_in_dataset


class VitDataset(pl.LightningDataModule):
    image_size: int = 224
    patch_size: int = 16
    n_classes: int = 1000

    def __init__(self, batch_size: int) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.proc_count = max(len(os.sched_getaffinity(0)), 32)
        logger.info(f"Using {self.proc_count} processes for data loading")


class Cifar10Dataset(VitDataset):

    patch_size = 4
    n_classes = 10

    COLUMNS = ["img", "label"]

    def _collate_fn(self, batch):
        images = []
        labels = []
        for sample in batch:
            images.append(to_tensor(sample["img"]))
            labels.append(sample["label"])
        return {"image": torch.stack(images), "label": torch.tensor(labels)}

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/cifar10"
        self.dataset = load_dataset("uoft-cs/cifar10")
        self.train_loader = DataLoader(self.dataset["train"], batch_size=self.batch_size, num_workers=self.proc_count, collate_fn=self._collate_fn)  # type: ignore
        self.val_loader = DataLoader(self.dataset["test"], batch_size=self.batch_size, num_workers=self.proc_count, collate_fn=self._collate_fn)  # type: ignore

        ensure_directory(cache_dir, clear=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


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
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
            ]
        )
        shard_list = get_wds_file_list("/weka/home-brianf/imagereward_cache")
        logger.info(f"Found {len(shard_list)} shards in dataset")
        self.train_wds_loader = wds.WebDataset(shard_list[:-1])
        self.val_wds_loader = wds.WebDataset(shard_list[-1])

    def train_dataloader(self):
        return DataLoader(
            self.train_wds_loader,
            batch_size=self.batch_size,
            num_workers=self.proc_count,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        # TODO fix train test split
        return DataLoader(self.val_wds_loader, num_workers=self.proc_count, batch_size=8, collate_fn=self.collate_fn)  # type: ignore

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Loading dataset for stage {stage}")

        cache_dir = "dataset_caches/image_reward"

        ensure_directory(cache_dir, clear=False)

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
            image_stream = io.BytesIO(sample["png"])
            image_pil = Image.open(image_stream)
            image = to_tensor(image_pil)
            image = resize(image, [self.image_size])
            image = center_crop(image, [self.image_size, self.image_size])
            image = image.to(dtype=torch.float32)
            image_tensors.append(image)

            json_dict = json.loads(sample["json"])
            labels.append(json_dict["overall_rating"])
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
