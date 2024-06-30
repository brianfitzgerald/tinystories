from transformers.optimization import get_cosine_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import LanguageModelHyperParams, SmModel
from torch.optim import AdamW

import lightning.pytorch as pl

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer


class LlamaFineTuner(SmModel):
    def __init__(self, params: LanguageModelHyperParams, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(params, tokenizer)
        self.params = params
        self.hparams.update(vars(params))

        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
            params.base_model_checkpoint
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(params.base_model_checkpoint)
        self.ckpt_name = params.base_model_checkpoint
        self.train_steps = 0
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels):
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    def _step(self, batch):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        print(f"Loss: {outputs.loss}")

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
