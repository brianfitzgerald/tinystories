import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import torch
from datasets import load_dataset
from loguru import logger
from peft.tuners.lora.config import LoraConfig
from peft.utils.constants import DUMMY_TARGET_MODULES
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import SchedulerType
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import PreferenceCollator
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from dataset.code import (
    CodeContestsDataModule,
    EvolCodeAlpacaDataModule,
)
from dataset.conversation import ConversationDataModule, ConversationRawDataModule
from model.utils import (
    DataModuleChoice,
    DatasetConfig,
    TuningModeChoice,
    ensure_directory,
    save_dataclass_to_json,
    short_hash,
)
from synthetic_data.utils import dictl
from trl_wrapper.dpo_trainer import CustomDPOTrainer, EvalDataModeChoice

MOCK_LLAMA = "qgallouedec/tiny-LlamaForCausalLM-3"
LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
QWEN_2_5_3B = "Qwen/Qwen2.5-3B-Instruct"
LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
SMOL_LM_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
# NOTE that mistral doesn't allow using system prompts, so it must be set to None.
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MINISTRAL_8B = "mistralai/Ministral-8B-Instruct-2410"


@dataclass
class WrapperConfig:
    model_id_or_path: str = LLAMA_3_2_1B
    notebook_mode: bool = False
    # Sequence length to trim completions to
    max_sequence_length: int = 1512
    max_prompt_length: int = 1024
    max_eval_sample_length: int = 1024
    max_samples: Optional[int] = None
    train_batch_size: int = 4
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    root_dir: Optional[str] = None
    data_module_choice: DataModuleChoice = "ultra_feedback"
    wandb_project_name: str = "codecontests-llama-3b"
    n_epochs: int = 1
    max_eval_dataset_size: Optional[int] = None
    # SFT only
    train_on_inputs: bool = True
    # adapter to load before training
    adapter_path: Optional[str] = None
    dpo_beta: float = 0.1
    learning_rate: float = 5e-5
    max_grad_norm: float = 0.3
    # lora config``
    lora_rank: int = 256
    lora_dropout: float = 0.05
    lora_alpha: int = 128
    logprob_precompute_batch_size: int = 16
    tuning_mode: TuningModeChoice = "dpo_full"
    eval_data_mode: EvalDataModeChoice = "random"
    eval_steps: int = 500
    save_steps: int = 1000
    using_mistral: bool = False
    run_suffix: Optional[str] = None
    special_tokens: Optional[List[str]] = None
    # Only used for Conversation dataset format
    input_dataset_name: Optional[str] = None
    custom_chat_template: Optional[str] = None
    lr_scheduler: SchedulerType = SchedulerType.CONSTANT
    neftune_noise_alpha: Optional[float] = None


LLAMA_CONFIG = WrapperConfig(
    model_id_or_path=LLAMA_3_2_1B,
    max_samples=20000,
    n_epochs=10,
    data_module_choice="ultra_feedback",
)

DOLPHIN_DPO_CONFIG = WrapperConfig(
    model_id_or_path=MISTRAL_7B,
    wandb_project_name="dolphin-dpo",
    train_batch_size=12,
    max_samples=20000,
    using_mistral=True,
)

CODECONTESTS_CONFIG = WrapperConfig(
    model_id_or_path=MISTRAL_7B,
    wandb_project_name="codecontests-ministral-8b",
    train_batch_size=12,
    data_module_choice="evol_codealpaca_dpo",
    using_mistral=True,
)

CODECONTESTS_SFT_CONFIG = WrapperConfig(
    model_id_or_path=MISTRAL_7B,
    wandb_project_name="codecontests-ministral-8b",
    train_batch_size=16,
    data_module_choice="evol_codealpaca_dpo",
    using_mistral=True,
    tuning_mode="sft",
    learning_rate=1e-5,
)

PLAYWRIGHT_CONFIG = WrapperConfig(
    model_id_or_path=MINISTRAL_8B,
    wandb_project_name="playwright",
    train_batch_size=4,
    data_module_choice="conversation",
    using_mistral=True,
    tuning_mode="sft",
    learning_rate=1e-5,
    run_suffix="cot",
    special_tokens=["<thought>", "</thought>", "<solution>", "</solution>"],
    input_dataset_name="screenplay_conversations.parquet",
    custom_chat_template="ministral_8b",
)

# llama 3 hparams
# https://huggingface.co/blog/llama3#fine-tuning-with-🤗-trl

CODECONTESTS_COT_CONFIG = WrapperConfig(
    model_id_or_path=LLAMA_3_2_3B,
    wandb_project_name="codecontests-ministral-8b",
    train_batch_size=4,
    data_module_choice="conversation_raw",
    tuning_mode="sft",
    gradient_checkpointing=False,
    learning_rate=1e-5,
    n_epochs=10,
    train_on_inputs=False,
    special_tokens=["<thought>", "</thought>", "<solution>", "</solution>"],
    custom_chat_template="llama3",
    input_dataset_name="openo1_sft_formatted_thoughts_conversations.parquet",
    neftune_noise_alpha=5,
    lr_scheduler=SchedulerType.COSINE,
)


CONFIGS = {
    "llama": LLAMA_CONFIG,
    "dolphin": DOLPHIN_DPO_CONFIG,
    "codecontests": CODECONTESTS_CONFIG,
    "codecontests_sft": CODECONTESTS_SFT_CONFIG,
    "codecontests_cot_sft": CODECONTESTS_COT_CONFIG,
    "playwright": PLAYWRIGHT_CONFIG,
}


class TrainerWrapper:
    def __init__(self, config: WrapperConfig, use_wandb: bool = False) -> None:
        self.config = config
        self.use_wandb = use_wandb

        # Init tokenizer here so we can use it without loading the model
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id_or_path
        )  # type: ignore
        # https://github.com/huggingface/trl/issues/1311#issuecomment-2016614091
        # self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        if self.config.special_tokens is not None:
            logger.info(f"Adding special tokens: {self.config.special_tokens}")
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": self.config.special_tokens}  # type: ignore
            )  # type: ignore

    def init_model(self):
        bnb_config = None

        if self.config.tuning_mode == "lora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info(f"Loading model {self.config.model_id_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id_or_path,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            ignore_mismatched_sizes=True,
            use_cache=not self.config.using_mistral,
        )
        if self.config.special_tokens is not None:
            logger.info(f"Resizing token embeddings for model to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

    def init_data_module(self):
        dataset_config = DatasetConfig(
            input_dataset_name=self.config.input_dataset_name,
            batch_size=self.config.train_batch_size,
            max_sequence_length=self.config.max_sequence_length,
            using_mistral=self.config.using_mistral,
            notebook_mode=self.config.notebook_mode,
            tuning_mode=self.config.tuning_mode,
            max_samples=self.config.max_samples,
            custom_chat_template=self.config.custom_chat_template,
            train_on_inputs=self.config.train_on_inputs,
        )
        if self.config.data_module_choice == "code_contests":
            self.data_module = CodeContestsDataModule(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "evol_codealpaca_dpo":
            self.data_module = EvolCodeAlpacaDataModule(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "conversation":
            self.data_module = ConversationDataModule(self.tokenizer, dataset_config)
        elif self.config.data_module_choice == "conversation_raw":
            self.data_module = ConversationRawDataModule(self.tokenizer, dataset_config)
        self.data_module.setup("fit")

    def init_trainer(self, comment: Optional[str] = None):
        # Get run name
        simple_date = datetime.now().strftime("%m-%d-%-H-%-M")
        random_id = int(torch.rand(1) * 1000000)
        model_id_without_org = self.config.model_id_or_path.split("/")[-1].lower()
        run_name = f"{simple_date}-{random_id}-{model_id_without_org}"
        if self.config.run_suffix is not None:
            run_name += f"-{self.config.run_suffix}"
        if comment is not None:
            run_name += f"-{comment}"

        output_dir = f"/weka/home-brianf/runs/{run_name}"
        logger.info(f"Saving output to: {output_dir}")

        os.environ["WANDB_PROJECT"] = self.config.wandb_project_name

        # LoRA config
        peft_config = None
        if self.config.tuning_mode in ("dpo_lora", "sft_lora"):
            peft_config = LoraConfig(
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                r=self.config.lora_rank,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            if self.model is not None and hasattr(self.model, "peft_config"):
                logger.info("LoRA already loaded, ignoring")
                peft_config.target_modules = DUMMY_TARGET_MODULES

        ensure_directory(output_dir)
        save_dataclass_to_json(self.config, f"{output_dir}/wrapper_config.json")
        model_id_hash = short_hash(self.config.model_id_or_path)
        self.ref_logpbrobs_cache_location = (
            f"{self.data_module.cache_dir}/{model_id_hash}/ref_logprobs_cache"
        )
        logger.info(
            f"Initializing trainer, run_name: {run_name}, wandb project: {self.config.wandb_project_name}"
        )
        logger.info(
            f"logprobs cache location: {self.ref_logpbrobs_cache_location} peft config: {peft_config is not None}"
        )
        logger.info(self.config)

        if self.config.tuning_mode in ("sft", "sft_lora"):
            args = SFTConfig(
                num_train_epochs=self.config.n_epochs,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_checkpointing=self.config.gradient_checkpointing,
                optim="adamw_torch_fused",
                learning_rate=self.config.learning_rate,
                max_grad_norm=self.config.max_grad_norm,
                warmup_ratio=0.1,
                lr_scheduler_type=self.config.lr_scheduler.value,
                logging_steps=10,
                save_steps=self.config.save_steps,
                save_total_limit=2,
                eval_strategy="steps",
                eval_on_start=True,
                eval_steps=self.config.eval_steps,
                bf16=True,
                tf32=True,
                push_to_hub=False,
                report_to="wandb" if self.use_wandb else "none",
                dataloader_num_workers=0 if self.config.notebook_mode else 4,
                dataset_num_proc=1 if self.config.notebook_mode else 4,
                max_seq_length=self.config.max_sequence_length,
                dataloader_pin_memory=True,
                run_name=run_name,
                dataset_text_field="conversation",
                output_dir=output_dir,
                disable_tqdm=not self.config.notebook_mode,
                neftune_noise_alpha=self.config.neftune_noise_alpha,
                use_liger=True
            )

            self.trainer = SFTTrainer(
                self.model,
                peft_config=peft_config,
                args=args,
                train_dataset=self.data_module.train_dataset,
                eval_dataset=self.data_module.val_dataset,
                tokenizer=self.tokenizer,  # type: ignore
            )
        else:
            args = DPOConfig(
                num_train_epochs=self.config.n_epochs,
                per_device_train_batch_size=self.config.train_batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_checkpointing=self.config.gradient_checkpointing,
                optim="adamw_torch_fused",
                learning_rate=self.config.learning_rate,
                max_grad_norm=self.config.max_grad_norm,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                logging_steps=1,
                save_steps=self.config.save_steps,
                save_total_limit=2,
                eval_strategy="steps",
                eval_on_start=True,
                eval_steps=self.config.eval_steps,
                bf16=True,
                tf32=False,
                push_to_hub=False,
                report_to="wandb" if self.use_wandb else "none",
                dataloader_num_workers=0 if self.config.notebook_mode else 4,
                dataset_num_proc=1 if self.config.notebook_mode else 4,
                max_length=self.config.max_sequence_length,
                max_prompt_length=self.config.max_prompt_length,
                precompute_ref_log_probs=not self.config.using_mistral,
                precompute_ref_batch_size=self.config.logprob_precompute_batch_size,
                dataloader_pin_memory=True,
                beta=self.config.dpo_beta,
                loss_type="sigmoid",
                generate_during_eval=True,
                run_name=run_name,
                output_dir=output_dir,
                disable_tqdm=not self.config.notebook_mode,
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
            self.trainer.set_custom_args(
                self.config.max_eval_sample_length,
                True,
                output_dir,
                self.config.eval_data_mode,
                self.config.using_mistral,
            )

            if self.trainer.precompute_ref_log_probs:
                eval_cache_location, train_cache_location = (
                    f"{self.ref_logpbrobs_cache_location}_eval.parquet",
                    f"{self.ref_logpbrobs_cache_location}_train.parquet",
                )
                if os.path.exists(train_cache_location):
                    logger.info("Loading cached logprobs...")
                    # TODO add support for eval dataset
                    self.trainer.train_dataset = load_dataset(
                        "parquet", data_files={"train": train_cache_location}
                    )["train"]  # type: ignore
                    self.trainer.eval_dataset = load_dataset(
                        "parquet", data_files={"train": eval_cache_location}
                    )["train"]  # type: ignore
                    self.trainer._precomputed_train_ref_log_probs = True
                    self.trainer._precomputed_eval_ref_log_probs = True
                    logger.info("Loaded.")
                else:
                    # force precomputing of reference logprobs
                    logger.info(
                        f"Precomputing reference logprobs, batch size: {self.config.logprob_precompute_batch_size}"
                    )
                    self.trainer.args.per_device_train_batch_size = (
                        self.config.logprob_precompute_batch_size
                    )

                    logger.info("Precomputing train logprobs")
                    self.trainer.get_train_dataloader()
                    logger.info("Precomputing eval logprobs")
                    self.trainer.get_eval_dataloader()
                    logger.info("Saving reference logprobs...")
                    assert self.trainer.eval_dataset is not None
                    assert self.trainer.train_dataset is not None

                    self.trainer.train_dataset.to_parquet(train_cache_location)
                    self.trainer.eval_dataset.to_parquet(eval_cache_location)
                    self.trainer.args.per_device_train_batch_size = (
                        self.config.train_batch_size
                    )

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
        logger.info("Starting training.")
        self.trainer.train()
