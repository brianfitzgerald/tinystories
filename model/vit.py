from dataclasses import dataclass
from typing import Literal, Optional

import lightning.pytorch as pl
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.aesthetic_score import VitDataset

"""
TODO: Use flash attention
TODO: implement simple vit
TODO: relative pos embedding

"""
OptimizerChoice = Literal["AdamW", "Adafactor", "AdamW8bit"]


@dataclass
class VitHParams:
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
    hidden_dim: int = 768
    depth: int = 12
    n_heads: int = 12
    mlp_dim: int = 3072
    pool: str = "cls"
    channels: int = 3
    head_dim: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        # scale factor for dot product
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # q * k^T
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # softmax(q * k^T)
        attn = self.softmax(dots)
        attn = self.dropout(attn)

        # v * softmax(q * k^T)
        out = torch.matmul(attn, v)

        # separate heads
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:  # type: ignore
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool="cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # Patchify image operation
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # absolute position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img: Tensor) -> Tensor:
        x: Tensor = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # repeat the cls token for each image in the batch
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # add the pos embedding to the patches
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class VisionTransformer(pl.LightningModule):
    def __init__(self, params: VitHParams, dataset: VitDataset):
        super().__init__()

        self.model = ViT(
            dataset.image_size,
            dataset.patch_size,
            dataset.n_classes,
            params.hidden_dim,
            params.depth,
            params.n_heads,
            params.mlp_dim,
            params.pool,
            params.channels,
            params.head_dim,
            params.dropout,
            params.emb_dropout,
        )

        self.params = params

        self.loss_fn = nn.CrossEntropyLoss()
        self.image_size = dataset.image_size

    def _step(self, batch: dict):
        image = batch["image"]
        pred = self.model(image)
        score = batch["label"]
        loss = self.loss_fn(pred, score)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            eps=self.params.adam_epsilon,
        )
        scheduler = CosineAnnealingLR(optimizer, self.params.num_train_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
