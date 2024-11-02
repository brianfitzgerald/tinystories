import torch
from torch import nn, Tensor
from dataclasses import dataclass
import math
import torch.nn.functional as F
from typing import Tuple
from torch.optim import AdamW
import bitsandbytes as bnb
from transformers.optimization import (
    get_linear_schedule_with_warmup,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from torchmetrics.text.perplexity import Perplexity
from tokenizers import normalizers, Regex


from model.utils import LMHyperParams, SmModel


@dataclass
class SimpleBERTConfig:
    hidden_size: int
    num_attention_heads: int
    pos_emb_radius: int
    embed_size: int
    num_layers: int


class SelfAttention(nn.Module):
    def __init__(self, config: SimpleBERTConfig) -> None:
        """
        Multi head self attention
        Uses pre-layernorm, and relative position embeddings
        No dropout for simplicity
        """
        super().__init__()

        self.config = config
        emb_size = config.embed_size
        n_head = config.num_attention_heads
        assert emb_size % n_head == 0

        # Rotational position embeddings
        pos_emb_radius = config.pos_emb_radius
        # split the embedding size into n_head parts
        pos_emb_units = config.embed_size // n_head

        # set as parameter so we can learn it
        self.pos_emb_k = nn.Parameter(torch.zeros(2 * pos_emb_radius, pos_emb_units))
        torch.nn.init.normal_(self.pos_emb_k, mean=0, std=0.02)

        # Disable bias - from cramming paper, removes unused parameters
        # qkv = 3 * emb_size
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)

        self.proj = nn.Linear(emb_size, emb_size, bias=False)

    def forward(self, x: Tensor):
        # x is context we are attending to
        # usually the hidden state of the transformer
        # the context is not divided by heads - each head attends to the whole context
        # the context is the output of the previous layer, not the qkv scores
        # TODO flash attention

        batch_size, context_size, emb_size = x.size()
        assert emb_size == self.config.embed_size

        n_head = self.config.num_attention_heads
        head_size = emb_size // n_head

        pos_emb_size, head_size = self.pos_emb_k.size()
        pos_emb_radius = self.config.pos_emb_radius
        assert pos_emb_size == 2 * pos_emb_radius

        # calculate in batch, and move the head to the batch dim
        # so out is (batch_size, n_head, context_size, head_size)
        k: Tensor = (
            self.key(x)
            .view(batch_size, context_size, n_head, head_size)
            .transpose(1, 2)
        )
        q: Tensor = (
            self.query(x)
            .view(batch_size, context_size, n_head, head_size)
            .transpose(1, 2)
        )
        v: Tensor = (
            self.value(x)
            .view(batch_size, context_size, n_head, head_size)
            .transpose(1, 2)
        )

        # Get the relative position embeddings for the context
        # i.e. matmul q and the position embeddings split out for each head
        # Only add position embeddings to keys, but not values
        k_pos_view = self.pos_emb_k.view(1, 1, pos_emb_size, head_size).transpose(
            -2, -1
        )
        att_rel_pos = q @ k_pos_view
        # create the relative position embeddings for the context
        # att_idxs = (context_size, context_size)
        att_idxs = (
            torch.clamp(
                torch.arange(context_size)[None, :]
                - torch.arange(context_size)[:, None],
                -pos_emb_radius,
                pos_emb_radius - 1,
            )
            % pos_emb_size
        ).to("cuda")

        # att_rel_pos = (batch_size, n_head, context_size, context_size)
        # this is the relative position embeddings for each head
        # gather retrieves the position embeddings for each value in k
        att_pos = torch.gather(
            att_rel_pos,
            3,
            att_idxs.expand((batch_size, n_head, context_size, context_size)),
        )
        assert att_pos.shape == (batch_size, n_head, context_size, context_size)

        attn_val = q @ k.transpose(-2, -1)

        # Scaling attention by the size of the key,
        # as per the original transformer paper
        # this helps keep the gradients from exploding
        attn_scale = 1.0 / math.sqrt(k.shape[-1])

        # Softmax over the last dim, and scale by the attention scale
        # attn_vals = (batch_size, n_head, context_size, context_size)
        att = F.softmax((attn_val + att_pos) * attn_scale, dim=-1)

        # perform the attention calculation
        # (batch_size, n_head, context_size, context_size) @ (batch_size, n_head, context_size, head_size)
        # returns (batch_size, n_head, context_size, head_size)
        y = att @ v
        # combine the heads
        y = y.transpose(1, 2).contiguous().view(batch_size, context_size, emb_size)

        y = self.proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: SimpleBERTConfig) -> None:
        super().__init__()

        embed_size = config.embed_size
        # eps - added to the variance in the normalization to avoid dividing by zero
        self.norm1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(embed_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size, bias=False),
        )
        self.pre_layernorm = True
        # No dropout layer, removed in Cramming

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_layernorm:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.mlp(x))

        return x


class BERT(nn.Module):
    def __init__(self, config: SimpleBERTConfig, vocab_size: int) -> None:
        super().__init__()

        self.config = config
        embed_size, n_layers = config.embed_size, config.num_layers

        self.token_emb = nn.Embedding(vocab_size, embed_size)
        # Layer norm for each embedding
        self.norm_emb = nn.LayerNorm(embed_size, eps=1e-6)

        self.transformer = nn.Sequential(*[Block(config) for _ in range(n_layers)])

        self.norm_final = nn.LayerNorm(embed_size, eps=1e-6)

        self.apply(self._init_weights)
        # apply the weight initialization to the projection layer, scaled by the number of layers
        # this is the same as the original transformer
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        """
        Initialize all Linear (attention and MLP) weights to normal(0, 0.02)
        and all biases to 0
        Initialize all Embedding weights to normal(0, 0.02)
        and all LayerNorms to bias of 0 and weight of 1
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_emb(x)
        x = self.norm_emb(x)

        x = self.transformer(x)

        # Added in Cramming
        x = self.norm_final(x)

        return x


class MLMHead(nn.Module):
    """
    Head for masked language modeling
    Cramming uses sparse prediction, but not implemented here
    Project the embeddings back to the vocab size, and calculate the loss

    # TODO - implement sparse prediction
    """

    def __init__(
        self, config: SimpleBERTConfig, vocab_size: int, ignore_token_index: int
    ) -> None:
        super().__init__()

        self.proj = nn.Linear(config.embed_size, vocab_size, bias=False)
        self.ignore_token_index = ignore_token_index

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        logits: Tensor = self.proj(x)
        # ignore_index is the padding token
        # the view is to flatten the logits and labels across the batch
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=self.ignore_token_index,
        )
        return logits, loss


class SimpleBertForMaskedLM(SmModel):
    """
    BERT model with a language modeling head
    """

    def __init__(self, hparams: LMHyperParams, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(hparams, tokenizer)

        config = SimpleBERTConfig(
            hidden_size=128,
            pos_emb_radius=16,
            embed_size=768,
            num_attention_heads=12,
            num_layers=12,
        )

        vocab_size = self.tokenizer.vocab_size
        ignore_token_index: int = self.tokenizer.pad_token_id  # type: ignore
        print(f"Ignore token index: {ignore_token_index}")
        self.perplexity = Perplexity(ignore_index=self.tokenizer.pad_token_id)

        self.bert = BERT(config, vocab_size)
        self.mlm_head = MLMHead(config, vocab_size, ignore_token_index)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.bert(x)
        logits, loss = self.mlm_head(x, y)
        perplexity = self.perplexity(logits, y)
        return logits, loss, perplexity

    def training_step(self, batch, batch_idx):
        logits, loss, perplexity = self(batch["input_ids"], batch["labels"])
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_ppl", perplexity, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        logits, loss, perplexity = self(batch["input_ids"], batch["labels"])
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("val_ppl", perplexity, on_step=True, on_epoch=True, logger=True)
        return {"val_loss": loss, "logits": logits}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        all_params = list(self.bert.parameters()) + list(self.mlm_head.parameters())

        # emulates the original optimizer in https://github.com/google-research/bert/blob/master/optimization.py#L65
        optimizer_grouped_parameters = [
            {
                "params": [p for p in all_params if p.dim() >= 2],
                "weight_decay": self.params.weight_decay,
            },
            {
                "params": [p for p in all_params if p.dim() < 2],
                "weight_decay": 0,
            },
        ]

        optim_choice = self.params.optimizer

        optimizer = None
        betas = (0.9, 0.999)

        if optim_choice == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.params.learning_rate,
                eps=self.params.adam_epsilon,
                betas=betas,
            )
        elif optim_choice == "AdamW8bit":
            optimizer = bnb.optim.adamw.AdamW8bit(
                optimizer_grouped_parameters,
                lr=self.params.learning_rate,
                eps=self.params.adam_epsilon,
                betas=betas,
            )
        else:
            raise ValueError(f"Unknown optimizer choice: {optim_choice}")

        warmup_steps = self.params.warmup_steps(self.trainer.estimated_stepping_batches)
        train_steps = self.trainer.num_training_batches
        print(f"Training steps: {train_steps} warmup steps: {warmup_steps}")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# copied from cramming/data/tokenizer_preparation.py in cramming-bert
def get_sane_normalizers(
    force_english_keyboard=False,
    force_lowercase=False,
    strip_accents=False,
    sanity=False,
):
    """original rules as in XLNET with optional modifications. force_english_keyboard is actually an ascii normalization."""
    if sanity:
        return normalizers.BertNormalizer(lowercase=force_lowercase)
    normalize_ops = []
    normalize_ops.append(normalizers.Replace("``", '"'))
    normalize_ops.append(normalizers.Replace("''", '"'))
    normalize_ops.append(normalizers.NFD() if strip_accents else normalizers.NFKC())
    if force_lowercase:
        normalize_ops.append(normalizers.Lowercase())
    if strip_accents:
        normalize_ops.append(normalizers.StripAccents())
    normalize_ops.append(normalizers.Replace(Regex(" {2,}"), " "))
    if force_english_keyboard:
        normalize_ops.append(
            normalizers.Replace(Regex(r"[^\x00-\x7F]+"), "")
        )  # start from 00 instead of 1F to include tab
    return normalizers.Sequence(normalize_ops)  # type: ignore
