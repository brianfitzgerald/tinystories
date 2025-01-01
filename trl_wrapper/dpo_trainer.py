from trl import DPOTrainer
import torch

import random
from typing import List, Optional
import wandb
from loguru import logger
from typing import Dict, List, Optional, Tuple, Literal
from contextlib import nullcontext
from torch.amp.autocast_mode import autocast
from transformers.generation.configuration_utils import GenerationConfig

from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer import Trainer
from trl.trainer.utils import pad_to_length
from tabulate import tabulate
import pandas as pd

from synthetic_data.utils import clean_message

EvalDataModeChoice = Literal["random", "fixed"]


class CustomDPOTrainer(DPOTrainer):
    def set_custom_args(
        self,
        max_eval_sample_length: int,
        eval_skip_special_tokens: bool,
        output_dir: str,
        eval_data_mode: EvalDataModeChoice,
        using_mistral: bool,
    ):
        """
        Set custom arguments needed for new functionality.
        This is so we don't have to modify the __init__ method of the Trainer class.
        """
        self.all_eval_rows = []
        self.max_eval_sample_length = max_eval_sample_length
        self.eval_skip_special_tokens = eval_skip_special_tokens
        self.output_dir = output_dir
        self.eval_data_mode = eval_data_mode
        self.using_mistral = using_mistral

    def generate_from_model_and_ref(
        self, model, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        generation_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=self.max_eval_sample_length,
            max_time=20,
            # https://github.com/huggingface/trl/issues/1217
            use_cache=not self.using_mistral,
        )

        with generate_context_manager:
            logger.info(
                f"Generating policy samples, max length: {self.max_eval_sample_length}..."
            )
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                pad_token_id=self.processing_class.pad_token_id,
                generation_config=generation_config,
            )

            # TODO cache the ref output samples?
            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        logger.info(
                            f"Generating reference samples, max length: {self.max_eval_sample_length}..."
                        )
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            pad_token_id=self.processing_class.pad_token_id,
                            generation_config=generation_config,
                        )
                else:
                    logger.info(
                        f"Generating reference samples, max length: {self.max_eval_sample_length}..."
                    )
                    assert self.ref_model is not None
                    ref_output = self.ref_model.generate(  # type: ignore
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        pad_token_id=self.processing_class.pad_token_id,
                        generation_config=generation_config,
                    )

        assert self.max_length is not None
        policy_output = pad_to_length(
            policy_output, self.max_length, self.processing_class.pad_token_id
        )
        policy_output_decoded = self.processing_class.batch_decode(
            policy_output, skip_special_tokens=self.eval_skip_special_tokens
        )

        ref_output = pad_to_length(
            ref_output, self.max_length, self.processing_class.pad_token_id
        )
        ref_output_decoded = self.processing_class.batch_decode(
            ref_output, skip_special_tokens=self.eval_skip_special_tokens
        )

        return policy_output_decoded, ref_output_decoded

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)  # type: ignore

            if self.eval_data_mode == "fixed":
                eval_indices = list(range(self.args.eval_batch_size))
            else:
                eval_indices = random.sample(
                    range(num_samples), k=self.args.eval_batch_size
                )

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(eval_indices)  # type: ignore
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            logger.info("Generating samples...")
            policy_output_decoded, ref_output_decoded = (
                self.generate_from_model_and_ref(self.model, random_batch)
            )  # type: ignore
            logger.info("Generated samples.")
            prompt_decoded = self.processing_class.batch_decode(
                random_batch["prompt_input_ids"],
                skip_special_tokens=self.eval_skip_special_tokens,
            )

            chosen_completion_decoded = self.processing_class.batch_decode(
                random_batch["chosen_input_ids"],
                skip_special_tokens=self.eval_skip_special_tokens,
            )
            rejected_completion_decoded = self.processing_class.batch_decode(
                random_batch["rejected_input_ids"],
                skip_special_tokens=self.eval_skip_special_tokens,
            )

            prefix = len(prompt_decoded)

            new_rows_to_log = []

            for i in range(len(prompt_decoded)):
                prefix = len(prompt_decoded[i])
                new_row_dict = {
                    "prompt": prompt_decoded[i],
                    "policy": policy_output_decoded[i][prefix:],
                    "ref": ref_output_decoded[i][prefix:],
                    "chosen": chosen_completion_decoded[i],
                    "rejected": rejected_completion_decoded[i],
                }
                new_row_dict = {k: clean_message(v) for k, v in new_row_dict.items()}
                new_rows_to_log.append(new_row_dict)

            new_rows_wandb_format = [k.values() for k in new_rows_to_log]
            wandb_headers = list(new_rows_to_log[0].keys())

            # TODO log to tabulate table if not using wandb, and save to txt file
            self.log(
                {
                    "eval_samples": wandb.Table(
                        columns=wandb_headers,
                        rows=new_rows_wandb_format,
                    )  # type: ignore
                }
            )

            self.all_eval_rows.extend(new_rows_to_log)

            all_rows_pd = pd.DataFrame(self.all_eval_rows)
            all_rows_pd.to_parquet(f"{self.output_dir}/eval_samples.parquet")

            tabulate_str = tabulate(
                new_rows_to_log,
                headers="keys",
                tablefmt="simple_grid",
                maxcolwidths=[50] * 5,
            )
            print(tabulate_str)
            current_step = self.state.global_step
            with open(f"{self.output_dir}/eval_samples.txt", "a") as f:
                f.write("\n" * 2)
                f.write(
                    "".join(
                        ["#" for _ in range(40)]
                        + [f"  Eval samples for step: {current_step}  "]
                        + ["#" for _ in range(40)]
                        + ["\n" * 2]
                    )
                )
                f.write(tabulate_str)

            logger.info("Saved eval samples.")

            self.state.log_history.pop()

        # Base evaluation
        initial_output = Trainer.evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return initial_output
