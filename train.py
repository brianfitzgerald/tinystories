import copy
from enum import IntEnum
from pprint import pprint
from typing import Dict, Sequence
from torch.utils.data import DataLoader
import evaluate
import torch
from dataclasses import dataclass
from evaluate import load
import torch.nn.functional as F
import os
import torch.nn as nn

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/"

from transformers import (
    GPTNeoConfig,
    PreTrainedTokenizer,
    get_scheduler,
    AutoTokenizer,
    AdamW,
    GPTNeoForCausalLM,
)
from datasets import load_dataset
from utils import *
from tqdm.auto import tqdm
import wandb

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": ("### Input:\n{start}\n\n### Code:\n{code}\n\n### Output:"),
    "prompt_no_input": ("### Input:\n{start}\n\n### Output:"),
}


@dataclass
class TrainingArgs:
    task = Task.STATE_CHANGES
    num_epochs = 6
    batch_size = 48
    eval_interval_epoch = 1
    save_interval = 1000
    eval_interval_batch = 500
    use_wandb = False
    push_model = False
    model_name = "smolmodels-finetune-33m-state-changes"
    use_peft = False
    seq_max_length = 80
    max_eval_batches = 5


if TrainingArgs.use_wandb:
    wandb.login()
    wandb_run = wandb.init(project=TrainingArgs.model_name)


def calculate_bpc(model, evaluation_data):
    total_loss = 0.0
    total_characters = 0

    model.eval()

    with torch.no_grad():
        for input_seq, target_seq in evaluation_data:
            input_seq = torch.tensor(input_seq).unsqueeze(0)
            target_seq = torch.tensor(target_seq).unsqueeze(0)

            output_seq = model(input_seq)
            output_seq = output_seq.squeeze(0)

            loss = F.cross_entropy(output_seq, target_seq)
            total_loss += loss.item()
            total_characters += target_seq.size(1)

    average_loss = total_loss / total_characters
    bpc = average_loss / torch.log(torch.tensor(2.0))

    return bpc.item()


tokenize_fn = None

if TrainingArgs.task == Task.TINY_STORIES:
    dataset = load_dataset("roneneldan/TinyStories")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenize_fn = lambda x: tokenizer(x["text"])

elif TrainingArgs.task == Task.STATE_CHANGES:
    dataset = load_dataset(
        "Fraser/python-state-changes", split="train", data_dir="mini", streaming=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")

    def format_state_changes(input_dict: dict):
        template = (
            PROMPT_DICT["prompt_input"]
            if input_dict.get("code", "") != ""
            else PROMPT_DICT["prompt_no_input"]
        )
        formatted_prompt = template.format_map(input_dict)
        output = f"{input_dict['end']}{tokenizer.eos_token}"
        composite = f"{formatted_prompt}{output}"
        return {"prompt": composite}

    def tokenize_state_changes(batch: dict) -> Dict:
        tokenized = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding="longest",
            max_length=TrainingArgs.seq_max_length,
            truncation=True,
        )
        return tokenized

    tokenize_fn = tokenize_state_changes
    dataset = dataset.map(format_state_changes, remove_columns=["start", "code", "end"])


seed, buffer_size = 42, 100_000
dataset = dataset.shuffle(seed, buffer_size=buffer_size)

dataset = dataset.with_format("torch")
dataset = dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=TrainingArgs.batch_size,
    remove_columns=["prompt"]
)

split_idx = int(buffer_size * 0.99)
train_dataset = dataset.take(split_idx)
eval_dataset = dataset.skip(split_idx)

train_dataloader = DataLoader(train_dataset, batch_size=TrainingArgs.batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=TrainingArgs.batch_size)


config = GPTNeoConfig(
    hidden_size=768,
    embed_dropout=0,
    attention_dropout=0,
    resid_dropout=0,
    max_position_embeddings=1024,
    num_heads=12,
    num_layers=6,
    attention_types=[[["global", "local"], 3]],
    window_size=256,
    layer_norm_epsilon=1e-5,
)

model = GPTNeoForCausalLM(config)
model.generation_config.max_new_tokens = TrainingArgs.seq_max_length
optimizer = AdamW(model.parameters(), lr=5e-5)

if TrainingArgs.use_wandb:
    wandb.watch(model)

num_training_steps = TrainingArgs.num_epochs * buffer_size
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = get_available_device()
model.to(device)

loss_fn = nn.CrossEntropyLoss()

progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(TrainingArgs.num_epochs):
    for j, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits, input_ids = get_model_output(outputs, batch)
        loss = loss_fn(logits, input_ids)
        loss.backward()

        if TrainingArgs.use_wandb:
            wandb.log({"loss": loss.item()})

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if j % TrainingArgs.eval_interval_batch == 0:
            log_dict = get_text_sample(logits, input_ids, tokenizer)
            pprint(log_dict, indent=2)
            if TrainingArgs.use_wandb:
                wandb.log(log_dict)

        if j % TrainingArgs.save_interval == 0:
            save_file_path = os.path.join(
                "checkpoints", f"model_epoch_{epoch}_batch_{j}"
            )
            model.save_pretrained(save_file_path, safe_serialization=True)
            if TrainingArgs.push_model:
                model.push_to_hub(TrainingArgs.model_name)

    # model.eval()
    # print("Running eval..")
    # for i, batch in enumerate(eval_dataloader):
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     logits, input_ids = get_model_output(outputs, batch)
    #     perplexity_score = get_perplexity(logits, input_ids)
    #     log_dict = {
    #         "perplexity": perplexity_score,
    #     }
    #     pprint(log_dict, indent=2)
    #     if TrainingArgs.use_wandb:
    #         wandb.log(log_dict)
    # model.train()
