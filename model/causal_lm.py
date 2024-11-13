from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import LMHyperParams, SmModel, ModelChoice
from torch.optim import AdamW
import torch

from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import BitsAndBytesConfig


class AutoLMFineTuner(SmModel):
    def __init__(self, params: LMHyperParams) -> None:
        super().__init__(params)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            params.base_model_checkpoint, trust_remote_code=True
        )  # type: ignore
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(params.base_model_checkpoint) # type: ignore
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.params = params
        self.hparams.update(vars(params))
        self.model_choice = ModelChoice.CAUSAL_LM


        self.ckpt_name = params.base_model_checkpoint
        self.train_steps = 0
        self.save_hyperparameters()
        if "smollm" in params.base_model_checkpoint:
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        assert self.model.generation_config
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def forward(self, input_ids, attention_mask, labels):
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    def _step(self, batch: dict):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        return outputs.loss

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
