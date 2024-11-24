import torch
import sys
from model.utils import LMHyperParams, SmModel, ModelChoice
from synthetic_data.utils import dictl
from dataset.squad import UltraFeedbackDataModule
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft.tuners.lora.config import LoraConfig
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from typing import cast
from peft.peft_model import PeftModel
import gc
from torch.amp.autocast_mode import autocast
import fire
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
from trl.trainer.dpo_trainer import PreferenceCollator
from typing import Dict

MOCK_LLAMA = "qgallouedec/tiny-LlamaForCausalLM-3"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"

@dataclass
class RLConfig:
    model_id: str = LLAMA_3_2_1B
    single_process_mode: bool = False
    max_seq_length: int = 1512
    prompt_length: int = 1024
    max_samples = 1000


class TrainerWrapper:

    def __init__(self, config: RLConfig) -> None:
        self.config = config

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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
    
    def init_data_module(self):
        self.data_module = UltraFeedbackDataModule(2, self.tokenizer, self.config.max_seq_length, self.config.max_samples, False)
        if self.config.single_process_mode:
            self.data_module.num_workers = 1
        self.data_module.setup("fit")

    def init_trainer(self):

        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        
        n_workers = 0 if self.config.single_process_mode else 4

        args = DPOConfig(
            output_dir="../outputs",
            num_train_epochs=1,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=5e-5,
            max_grad_norm=0.3,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=700,
            bf16=True,
            tf32=True,
            push_to_hub=False,
            report_to="tensorboard",
            # debugger will fail without this
            dataloader_num_workers=n_workers,
            dataset_num_proc=1,
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.prompt_length,
            precompute_ref_log_probs=True,
            dataloader_pin_memory=True,
            beta=0.1,
            loss_type="sigmoid",
        )


        self.trainer = DPOTrainer(
            self.model,
            ref_model=None,  # set to none since we use peft
            peft_config=peft_config,
            args=args,
            train_dataset=self.data_module.train_dataset,
            eval_dataset=self.data_module.val_dataset,
            tokenizer=self.tokenizer,  # type: ignore
        )

    def compute_loss_metrics(self):
        assert self.trainer._peft_has_been_casted_to_bf16
        # precompute reference logprobs
        self.trainer.get_train_dataloader()
        collator = PreferenceCollator(self.tokenizer.pad_token_id, "pt")  # type: ignore
        batch_size = 1
        outputs = []
        with torch.no_grad(), autocast("cuda"):
            for batch in tqdm(
                self.trainer.train_dataset.iter(batch_size=batch_size),
                desc="Computing DPO loss",
                total=len(self.trainer.train_dataset) // batch_size,
            ):
                sample_collated = collator(dictl(batch))
                metrics = self.get_sample_wise_metrics(sample_collated)
                for i in range(batch_size):
                    out_sample = {
                        "prompt": batch["prompt"][i],
                        "chosen": batch["chosen"][i],
                        "rejected": batch["rejected"][i],
                    }
                    for k, v in metrics.items():
                        if isinstance(v, list):
                            out_sample[k] = v[i]
                        else:
                            out_sample[k] = v
                    outputs.append(out_sample)
        return outputs

    def get_sample_wise_metrics(self, batch: dict):
        metrics = {}

        model_output = self.trainer.concatenated_forward(self.model, batch) # type: ignore

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.trainer.compute_ref_log_probs(batch)

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
        }

        for k in [
            "chosen_logps",
            "rejected_logps",
        ]:
            metrics[k] = model_output[k].tolist()
        return metrics

    def train(self):
        logger.info("Training model")
        self.trainer.train()

def main():
    cfg = RLConfig()
    wrapper = TrainerWrapper(cfg)
    wrapper.init_model()
    wrapper.init_data_module()
    wrapper.init_trainer()

if __name__ == "__main__":
    fire.Fire(main)