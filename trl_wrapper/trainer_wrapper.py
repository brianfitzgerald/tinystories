import os
from dataclasses import dataclass
from typing import Literal, Optional

import fire
import pandas as pd
import torch
from datasets import load_dataset
from loguru import logger
from peft.tuners.lora.config import LoraConfig
from peft.utils.constants import DUMMY_TARGET_MODULES
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import PreferenceCollator

from dataset.squad import CodeContestsDataModule, UltraFeedbackDataModule
from synthetic_data.utils import dictl
from trl_wrapper.dpo_trainer import CustomDPOTrainer


MOCK_LLAMA = "qgallouedec/tiny-LlamaForCausalLM-3"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
SMOL_LM_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"

DataModuleChoice = Literal["ultra_feedback", "code_contests"]


@dataclass
class WrapperConfig:
    model_id: str = LLAMA_3_2_1B
    single_process_mode: bool = False
    max_seq_length: int = 1512
    prompt_length: int = 1024
    max_samples: Optional[int] = None
    train_batch_size: int = 4
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    using_filtered_logprobs: bool = False
    root_dir: Optional[str] = None
    data_module_choice: DataModuleChoice = "code_contests"
    wandb_project_name: str = "codecontests-llama-3b"
    n_epochs: int = 1
    max_eval_dataset_size: Optional[int] = None


LLAMA_CONFIG = WrapperConfig(
    model_id=LLAMA_3_2_1B, max_samples=10000, using_filtered_logprobs=True, n_epochs=10
)


class TrainerWrapper:

    def __init__(self, config: WrapperConfig, use_wandb: bool = False) -> None:
        self.config = config
        self.use_wandb = use_wandb

    def init_model(self):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logger.info(f"Loading model {self.config.model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.config.model_id)  # type: ignore
        # https://github.com/huggingface/trl/issues/1311#issuecomment-2016614091
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.model.resize_token_embeddings(len(self.tokenizer))
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

    def init_data_module(self):
        if self.config.data_module_choice == "ultra_feedback":
            self.data_module = UltraFeedbackDataModule(
                self.config.train_batch_size,
                self.tokenizer,
                self.config.max_seq_length,
                self.config.max_samples,
                self.config.single_process_mode,
                self.config.using_filtered_logprobs,
            )
        else:
            self.data_module = CodeContestsDataModule(
                self.config.train_batch_size,
                self.tokenizer,
                self.config.max_seq_length,
                self.config.max_eval_dataset_size,
            )
        if self.config.single_process_mode:
            self.data_module.num_workers = 1
        self.data_module.setup("fit")

    def init_trainer(self):

        peft_config = LoraConfig(
            lora_alpha=256,
            lora_dropout=0.05,
            r=512,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

        n_workers = 0 if self.config.single_process_mode else 4

        if self.model is not None and hasattr(self.model, "peft_config"):
            logger.info("LoRA already loaded")
            peft_config.target_modules = DUMMY_TARGET_MODULES

        random_run_name = f"run-{int(torch.rand(1) * 1000000)}"

        os.environ["WANDB_PROJECT"] = self.config.wandb_project_name

        args = DPOConfig(
            num_train_epochs=self.config.n_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=False,
            optim="adamw_bnb_8bit",
            learning_rate=5e-5,
            max_grad_norm=0.3,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=250,
            save_total_limit=2,
            eval_strategy="steps",
            eval_on_start=True,
            eval_steps=100,
            bf16=True,
            tf32=False,
            push_to_hub=False,
            report_to="wandb" if self.use_wandb else "none",
            # debugger will fail without this
            dataloader_num_workers=n_workers,
            dataset_num_proc=1,
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.prompt_length,
            precompute_ref_log_probs=True,
            dataloader_pin_memory=True,
            beta=0.1,
            loss_type="sigmoid",
            generate_during_eval=True,
            run_name=random_run_name,
            output_dir="../outputs",
        )

        self.ref_logpbrobs_cache_location = (
            f"{self.data_module.cache_dir}/dpo_computed_dataset.parquet"
        )

        logger.info(
            f"Initializing DPOTrainer, with project name: {self.config.wandb_project_name}, run_name: {random_run_name}"
        )

        self.trainer = CustomDPOTrainer(
            self.model,
            ref_model=None,  # set to none since we use peft
            peft_config=peft_config,
            args=args,
            train_dataset=self.data_module.train_dataset,
            eval_dataset=self.data_module.val_dataset,
            tokenizer=self.tokenizer,  # type: ignore
        )

        if self.trainer.precompute_ref_log_probs:
            if os.path.exists(self.ref_logpbrobs_cache_location):
                logger.info("Loading cached logprobs...")
                # TODO add support for eval dataset
                self.trainer.train_dataset = load_dataset("parquet", data_files={"train": self.ref_logpbrobs_cache_location})["train"]  # type: ignore
                self.trainer._precomputed_train_ref_log_probs = True
            else:
                # force precomputing of reference logprobs
                logger.info("Precomputing reference logprobs...")
                self.trainer.get_train_dataloader()
                logger.info("Saving reference logprobs...")
                self.trainer.train_dataset.to_parquet(self.ref_logpbrobs_cache_location)

    def compute_loss_metrics(self, batch_size: int = 1):
        """
        Iterate through all batches and compute metrics sample-wise.
        Keep on batch_size=1 unless needed
        """
        assert self.trainer._peft_has_been_casted_to_bf16
        # precompute reference logprobs
        self.trainer.get_train_dataloader()
        collator = PreferenceCollator(self.tokenizer.pad_token_id, "pt")  # type: ignore
        outputs = []
        with torch.no_grad(), autocast("cuda"):
            batch_iter = tqdm(
                self.trainer.train_dataset.iter(batch_size=batch_size),
                desc="Computing DPO loss",
                total=len(self.trainer.train_dataset) // batch_size,
            )
            for batch in batch_iter:
                sample_collated = collator(dictl(batch))
                metrics = self.get_sample_wise_metrics(sample_collated)
                for j in range(batch_size):
                    out_sample = {
                        "prompt": batch["prompt"][j],
                        "chosen": batch["chosen"][j],
                        "rejected": batch["rejected"][j],
                    }
                    for k, v in metrics.items():
                        if isinstance(v, list):
                            out_sample[k] = v[j]
                        else:
                            out_sample[k] = v
                    outputs.append(out_sample)
        return outputs

    def get_sample_wise_metrics(self, batch: dict):
        """
        Return sample-wise loss metrics for a single batch
        """
        metrics = {}

        model_output = self.trainer.concatenated_forward(self.model, batch)  # type: ignore

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.trainer.compute_ref_log_probs(
                batch
            )

        losses, chosen_rewards, rejected_rewards = self.trainer.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = chosen_rewards - rejected_rewards

        metrics = {
            "loss": losses.tolist(),
            "reward_accuracy": reward_accuracies.tolist(),
            "reward_margin": reward_margins.tolist(),
            "chosen_rewards": chosen_rewards.tolist(),
            "rejected_rewards": rejected_rewards.tolist(),
        }

        for k in [
            "chosen_logps",
            "rejected_logps",
            "mean_chosen_logits",
            "mean_rejected_logits",
        ]:
            metrics[k] = model_output[k].tolist()
        return metrics

    def train(self):
        logger.info("Training model")
        self.trainer.train()
