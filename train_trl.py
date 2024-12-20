import fire
from loguru import logger
from trl_wrapper.trainer_wrapper import (
    TrainerWrapper,
    CONFIGS,
)
from typing import Optional


def main(config: str = "dolphin", notebook_mode: bool = False, wandb: bool = False, comment: Optional[str] = None):

    cfg = CONFIGS[config]
    if notebook_mode:
        cfg.notebook_mode = True
    wrapper = TrainerWrapper(cfg, wandb)
    wrapper.init_model()
    wrapper.init_data_module()
    wrapper.init_trainer(comment)
    logger.info(f"Starting training, config: {config}")
    wrapper.train()


if __name__ == "__main__":
    fire.Fire(main)
