import fire
import pandas as pd
from loguru import logger
from trl_wrapper.trainer_wrapper import LLAMA_CONFIG, TrainerWrapper


def main(generate_logprobs: bool = False, wandb: bool = False):
    wrapper = TrainerWrapper(LLAMA_CONFIG, wandb)
    wrapper.init_model()
    wrapper.init_data_module()
    wrapper.init_trainer()

    if generate_logprobs:
        logger.info("Computing loss metrics...")
        outputs = wrapper.compute_loss_metrics()
        pd.DataFrame(outputs).to_parquet("scored_sorted_logprobs.parquet")
        return outputs
    else:
        logger.info("Starting training...")
        wrapper.train()


if __name__ == "__main__":
    fire.Fire(main)
