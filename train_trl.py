import fire
import pandas as pd
from loguru import logger
from trl_wrapper.trainer_wrapper import LLAMA_CONFIG, DOLPHIN_DPO_CONFIG, TrainerWrapper, CONFIGS


def main(config: str = "dolphin", notebook_mode: bool = False, wandb: bool = False):
    
    cfg = CONFIGS[config]
    if notebook_mode:
        cfg.notebook_mode = True
    wrapper = TrainerWrapper(DOLPHIN_DPO_CONFIG, wandb)
    wrapper.init_model()
    wrapper.init_data_module()
    wrapper.init_trainer()
    logger.info("Starting training...")
    wrapper.train()


if __name__ == "__main__":
    fire.Fire(main)
