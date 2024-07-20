from loguru import logger
from transformers.optimization import (
    Adafactor,
    get_inverse_sqrt_schedule,
    AdafactorSchedule,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from model.utils import LanguageModelHyperParams, IGNORE_TOKEN_INDEX, SmModel
from torch.optim import AdamW
from torch import Tensor
from torchmetrics.text.perplexity import Perplexity
import bitsandbytes as bnb


from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


class T5FineTuner(SmModel):
    def __init__(self, params: LanguageModelHyperParams, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(params, tokenizer)

        logger.info("Loading T5 model")
        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(params.base_model_checkpoint, device_map=0)
        ) # type: ignore
        self.train_steps = 0
        self.save_hyperparameters()
        self.perplexity = Perplexity(ignore_index=self.tokenizer.pad_token_id)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        decoder_attention_mask: Tensor,
        labels: Tensor,
    ):
        labels[labels[:, :] == self.tokenizer.pad_token_id] = IGNORE_TOKEN_INDEX
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        perplexity = self.perplexity(outputs.logits, batch["labels"])

        return outputs.loss, perplexity

    def training_step(self, batch, batch_idx):
        loss, perplexity = self._step(batch)
        self.log(
            "train_ppl",
            perplexity,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, perplexity = self._step(batch)
        self.log(
            "val_ppl",
            perplexity,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=False
        )
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        # emulates the original optimizer in https://github.com/google-research/bert/blob/master/optimization.py#L65
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.params.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optim_choice = self.params.optimizer

        if optim_choice in ["AdamW", "AdamW8bit"]:
            if optim_choice == "AdamW":
                optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.params.learning_rate,
                    eps=self.params.adam_epsilon,
                )
            elif optim_choice == "AdamW8bit":
                optimizer = bnb.optim.adamw.AdamW8bit(
                    optimizer_grouped_parameters,
                    lr=self.params.learning_rate,
                    eps=self.params.adam_epsilon,
                )
            scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=self.params.warmup_steps(
                    self.trainer.estimated_stepping_batches
                ),
            )
        elif optim_choice == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.params.learning_rate,
                eps=(self.params.adam_epsilon, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
            scheduler = AdafactorSchedule(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
