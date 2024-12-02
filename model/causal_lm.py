from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import LMHyperParams, SmModel, ModelChoice, TuningType
from torch.optim import AdamW
import torch.nn.functional as F
import torch
from torch import Tensor as T
from trl.models.modeling_base import create_reference_model
from loguru import logger

from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import BitsAndBytesConfig
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from typing import Tuple


class AutoLMFineTuner(SmModel):
    def __init__(self, params: LMHyperParams) -> None:
        super().__init__(params)
        self.model: PreTrainedModel | PeftModel = AutoModelForCausalLM.from_pretrained(
            params.base_model_checkpoint, trust_remote_code=True
        )  # type: ignore
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(params.base_model_checkpoint)  # type: ignore
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.params = params
        self.hparams.update(vars(params))
        self.model_choice = ModelChoice.CAUSAL_LM
        self.tuning_type: TuningType = params.tuning_type
        if params.tuning_type == "dpo":
            self.peft_config = LoraConfig(
                lora_alpha=128,
                lora_dropout=0.05,
                r=256,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            logger.info(f"Using DPO with LoraConfig: {self.peft_config}")
            self.reference_model = create_reference_model(self.model) # type: ignore
            self.model = get_peft_model(self.model, self.peft_config)  # type: ignore
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.ckpt_name = params.base_model_checkpoint
        self.train_steps = 0
        self.save_hyperparameters()
        if "smollm" in params.base_model_checkpoint:
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        assert self.model.generation_config
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask, labels):
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    def _step(self, batch: dict) -> T:
        if self.tuning_type == "sft":
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            return outputs.loss
        elif self.tuning_type == "dpo":
            model_out_dict = {}
            for label in ["chosen", "rejected"]:
                for model_name, model in [("reference", self.reference_model), ("policy", self.model)]:
                    model_out_dict[f"{model_name}_{label}"] = model(
                        input_ids=batch[f"{label}_input_ids"],
                        attention_mask=batch[f"{label}_attention_mask"],
                        labels=batch[f"{label}_labels"],
                    )
            # TODO log rewards
            losses, chosen_rewards, rejected_rewards = dpo_loss(
                model_out_dict["policy_chosen"].logits,
                model_out_dict["policy_rejected"].logits,
                model_out_dict["reference_chosen"].logits,
                model_out_dict["reference_rejected"].logits,
            )
            return losses.mean()
        else:
            raise ValueError(f"Invalid tuning type: {self.tuning_type}")

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
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            eps=self.params.adam_epsilon,
            weight_decay=self.params.weight_decay,
        )
        print(f"Configuring optimizers: {self.train_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.params.warmup_steps(
                self.trainer.estimated_stepping_batches
            ),
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
