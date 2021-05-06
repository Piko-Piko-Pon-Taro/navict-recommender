# -*- coding: utf-8 -*-
"""Main script
This module is for training.

Args:
    project (str): Name of project yaml file.
"""

import hydra
from omegaconf import DictConfig

from trainers import get_trainer


@hydra.main(config_path="configs/project", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main function
    
    Args:
        cfg: Hydra configs.
    
    Returns:
        None.
    """

    trainer = get_trainer(cfg)
    trainer.execute(eval=cfg.train.eval)


if __name__ == '__main__':
    main()