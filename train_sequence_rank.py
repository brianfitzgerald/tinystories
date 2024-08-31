import pandas as pd
from transformers import DebertaV2Model, DebertaV2Tokenizer
import torch
from torch import Tensor as T
from tqdm import tqdm
from torch.nn import BCELoss
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import datasets

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


tokenizer_kwargs = {
    "padding": "max_length",
    "max_length": 128,
    "truncation": True,
    "return_tensors": "pt",
}

TEXT_COLUMNS = ["prompt", "response_a", "response_b"]


def main(model_name="microsoft/deberta-v3-base", batch_size=16):

    train_df = pd.read_csv("data/train.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deberta_model = DebertaV2Model.from_pretrained(model_name).to(device)  # type: ignore
    tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    def tokenize_batch(batch: pd.DataFrame) -> EncodedBatch:
        batch_out = {feature: [] for feature in TEXT_COLUMNS}
        with torch.autocast(device_type=deberta_model.device.type):
            for feature in TEXT_COLUMNS:
                feat_list = batch[feature]
                feat_tokenized = tokenizer(feat_list, **tokenizer_kwargs)["input_ids"]  # type: ignore
                batch_out[feature].append(feat_tokenized)

        for feature in TEXT_COLUMNS:
            batch_out[feature] = torch.cat(batch_out[feature], dim=0)

        winners = []
        for i, winner_a in enumerate(batch["winner_model_a"]):
            winner = 1 if winner_a == 1 else 0
            winners.append(winner)

        batch_out["winners"] = torch.tensor(winners)

        return batch_out

    def model_step(batch: EncodedBatch) -> T:
        model_encodings: EncodedBatch = {}  # type: ignore
        for feat in TEXT_COLUMNS:
            model_outputs = deberta_model(**batch[feat])
            last_hidden_state: T = model_outputs.last_hidden_state
            model_encodings[feat] = last_hidden_state

        rankings = ranker(
            model_encodings["response_a"], model_encodings["response_b"]
        ).to(dtype=torch.float32)
        return rankings

    train_ds = datasets.Dataset.from_pandas(train_df)
    train_ds = train_ds.map(tokenize_batch, batched=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size)  # type: ignore

    pooler_hidden_size: int = deberta_model.config.hidden_size
    ranker = SequenceRanker(pooler_hidden_size)
    ranker = ranker.to(device)
    loss_fn = BCELoss()
    optim = Adam(
        [{"params": ranker.parameters()}, {"params": deberta_model.parameters()}],
        lr=1e-3,
    )

    writer = SummaryWriter()

    train_df, val_df = train_test_split(train_df, test_size=16)

    train_loss, val_loss = 0.0, 0.0

    for epoch in range(100):

        train_iter = tqdm(range(0, len(train_df), batch_size))
        for i, batch in enumerate(train_loader):
            if i >= len(train_df) - batch_size:
                break
            global_step = i + epoch * len(train_df)
            optim.zero_grad()
            rankings = model_step(batch)
            loss = loss_fn(rankings, encodings["winners"].to(dtype=torch.float32))
            loss.backward()
            train_loss = loss.item()
            optim.step()
            writer.add_scalar("train/loss", loss.item(), global_step)
            if i % 100 == 0:
                for j in range(0, len(val_df), batch_size):
                    batch = val_df.iloc[j : j + batch_size]
                    encodings = tokenize_batch(
                        batch, deberta_model, tokenizer, device, batch_size
                    )
                    rankings = ranker(
                        encodings["response_a"], encodings["response_b"]
                    ).to(dtype=torch.float32)
                    loss = loss_fn(
                        rankings, encodings["winners"].to(dtype=torch.float32)
                    )
                    val_loss = loss.item()
                    writer.add_scalar("val/loss", val_loss, global_step + j)
            train_iter.set_description(
                f"Epoch: {epoch} Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}"
            )


if __name__ == "__main__":
    main()
