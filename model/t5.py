import bitsandbytes as bnb
from loguru import logger
from torch import Tensor
from torch.optim import AdamW
from torchmetrics.functional import f1_score, precision, recall, perplexity
from torchmetrics.text.ter import TranslationEditRate
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.optimization import (
    Adafactor,
    get_inverse_sqrt_schedule,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Tuple, Dict

from model.utils import (
    IGNORE_TOKEN_INDEX,
    LMHyperParams,
    ModelChoice,
    SmModel,
)


class T5FineTuner(SmModel):
    def __init__(self, params: LMHyperParams, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(params, tokenizer)

        logger.info("Loading T5 model")
        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(params.base_model_checkpoint)
        )  # type: ignore
        self.train_steps = 0
        self.save_hyperparameters()
        self.ter = TranslationEditRate(ignore_index=self.tokenizer.pad_token_id)
        self.model_choice = ModelChoice.T5

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

    def _step(self, batch: Dict, split: str) -> Tuple[Tensor, Tensor]:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
        )
        self.log(
            f"{split}_ppl",
            perplexity(outputs.logits, outputs.labels),
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"{split}_ter",
            self.ter(outputs.logits, outputs.labels),
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"{split}_precision",
            precision(
                outputs.logits, outputs.labels, average="macro", task="multiclass"
            ),
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"{split}_recall",
            recall(outputs.logits, outputs.labels, average="macro", task="multiclass"),
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"{split}_f1",
            f1_score(
                outputs.logits, outputs.labels, average="macro", task="multiclass"
            ),
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        return outputs.loss

    def training_step(self, batch, batch_idx):
        # debug_tokens = self.tokenizer.decode(batch['input_ids'][0].tolist())
        # logger.debug(f"Debug input: {debug_tokens}")
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "validation")
        return loss

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        # emulates the original optimizer in https://github.com/google-research/bert/blob/master/optimization.py#L65
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_groups = [
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
                    optimizer_groups,
                    lr=self.params.learning_rate,
                    eps=self.params.adam_epsilon,
                )
            elif optim_choice == "AdamW8bit":
                optimizer = bnb.optim.adamw.AdamW8bit(
                    optimizer_groups,
                    lr=self.params.learning_rate,
                    eps=self.params.adam_epsilon,
                )
            scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=self.params.warmup_steps(
                    self.trainer.estimated_stepping_batches
                ),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        elif optim_choice == "Adafactor":
            optimizer = Adafactor(
                optimizer_groups,
                lr=self.params.learning_rate,
                eps=(self.params.adam_epsilon, 1e-3),
                relative_step=False,
                scale_parameter=False,
            )
            return {
                "optimizer": optimizer,
            }
