import pandas as pd
from transformers import DebertaV2Model, DebertaV2Tokenizer
import torch
from torch import Tensor as T
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from datasets import Dataset
import os

# https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview

import torch.nn.functional as F
from torch import nn
from typing import TypedDict, List
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout


class EncodedBatch(TypedDict):
    prompt: T
    response_a: T
    response_b: T
    winners: T


class ContextPooler(nn.Module):
    def __init__(self, hidden_size: int, pooler_dropout: float = 0):
        super(ContextPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = StableDropout(pooler_dropout)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: T) -> T:
        # get the hidden_state of the first token
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = self.act_fn(pooled_output)

        return pooled_output


class SequenceRanker(nn.Module):
    def __init__(
        self,
        pooler_hidden_size: int,
        pooler_dropout: float = 0,
        ranker_layer_sizes: List[int] = [512, 256, 128],
    ):
        super(SequenceRanker, self).__init__()
        self.pooler = ContextPooler(pooler_hidden_size, pooler_dropout)

        ranker_layers: List[nn.Module] = [
            nn.Linear(pooler_hidden_size, ranker_layer_sizes[0])
        ]
        for _, (in_size, out_size) in enumerate(
            zip(ranker_layer_sizes[:-1], ranker_layer_sizes[1:])
        ):
            ranker_layers.append(nn.Linear(in_size, out_size))
            ranker_layers.append(nn.BatchNorm1d(out_size))
            ranker_layers.append(nn.ReLU())
            ranker_layers.append(nn.Dropout(0.1))

        ranker_layers.append(nn.Linear(ranker_layer_sizes[-1], 1))
        self.ranker = nn.Sequential(*ranker_layers)

    def forward(self, hidden_states_a: T, hidden_states_b: T) -> T:
        out_a = self.pooler(hidden_states_a)
        out_a = self.ranker(out_a)

        out_b = self.pooler(hidden_states_b)
        out_b = self.ranker(out_b)

        # output probability that a is greater than b
        out = F.sigmoid(out_a - out_b)
        return out.squeeze()


RESPONSE_COLUMNS = ["response_a", "response_b"]
TEXT_COLUMNS = ["prompt"] + RESPONSE_COLUMNS
TOKENIZED_COLUMNS = (
    [f"{col}_input_ids" for col in TEXT_COLUMNS]
    + [f"{col}_attention_mask" for col in TEXT_COLUMNS]
    + ["winners"]
)

tokenizer_kwargs = {
    "padding": "max_length",
    "max_length": 256,
    "truncation": True,
    "return_tensors": "pt",
}


def main(model_name="microsoft/deberta-v3-base", batch_size=64):
    input_df = pd.read_csv("data/train.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deberta_model = DebertaV2Model.from_pretrained(model_name).to(device)  # type: ignore
    tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    os.makedirs("dataset_caches/sequence_rank", exist_ok=True)

    def tokenize_batch(batch: dict) -> dict:
        batch_out = {}
        for feature in TEXT_COLUMNS:
            feat_list = batch[feature]
            feat_tokenized = tokenizer(feat_list, **tokenizer_kwargs)
            batch_out[f"{feature}_input_ids"] = feat_tokenized["input_ids"]
            batch_out[f"{feature}_attention_mask"] = feat_tokenized["attention_mask"]

        winners = []
        for winner_a in batch["winner_model_a"]:
            winner = 1 if winner_a == 1 else 0
            winners.append(winner)

        batch_out["winners"] = torch.tensor(winners)  # type: ignore

        return batch_out

    def model_step(batch: EncodedBatch) -> T:
        model_encodings: EncodedBatch = {}  # type: ignore
        for feature in RESPONSE_COLUMNS:
            model_outputs = deberta_model(
                input_ids=batch[f"{feature}_input_ids"].to(device),
                attention_mask=batch[f"{feature}_attention_mask"].to(device),
            )
            last_hidden_state: T = model_outputs.last_hidden_state
            model_encodings[feature] = last_hidden_state

        rankings = ranker(
            model_encodings["response_a"], model_encodings["response_b"]
        ).to(dtype=torch.float32)
        return rankings

    dataset = Dataset.from_pandas(input_df)
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        cache_file_name="dataset_caches/sequence_rank/cache.parquet",
        load_from_cache_file=True,
        num_proc=4,
        batch_size=128,
    )
    dataset.set_format(type="torch", columns=TOKENIZED_COLUMNS)
    dataset = dataset.select_columns(TOKENIZED_COLUMNS)

    dataset = dataset.train_test_split(test_size=0.05)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, drop_last=True)  # type: ignore
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, drop_last=True)  # type: ignore

    pooler_hidden_size: int = deberta_model.config.hidden_size
    ranker = SequenceRanker(pooler_hidden_size)
    ranker = ranker.to(device)
    loss_fn = BCELoss()
    optim = Adam(
        [{"params": ranker.parameters()}, {"params": deberta_model.parameters()}],
        lr=1e-3,
    )

    writer = SummaryWriter()

    train_loss, val_loss = 0.0, 0.0

    def evaluate():
        test_iter = tqdm(
            enumerate(test_loader), total=len(dataset["test"]) // batch_size, desc="Val"
        )
        for j, val_batch in test_iter:
            with torch.no_grad():
                rankings = model_step(val_batch)
                loss = loss_fn(
                    rankings,
                    batch["winners"].to(dtype=torch.float32, device=device),
                )
                val_loss = loss.item()
                writer.add_scalar("val/loss", val_loss, global_step + j)

    for epoch in range(100):
        train_iter = tqdm(
            enumerate(train_loader), total=len(dataset["train"]) // batch_size
        )
        for i, batch in train_iter:
            global_step = i + (epoch * len(train_loader))
            optim.zero_grad()
            rankings = model_step(batch)
            loss = loss_fn(
                rankings, batch["winners"].to(dtype=torch.float32, device=device)
            )
            loss.backward()
            train_loss = loss.item()
            optim.step()
            writer.add_scalar("train/loss", loss.item(), global_step)
            if i % 500 == 0:
                evaluate()

        # eval at end of every batch
        evaluate()

    train_iter.set_description(
        f"Epoch: {epoch} Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
