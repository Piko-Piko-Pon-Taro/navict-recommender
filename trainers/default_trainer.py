# -*- coding: utf-8 -*-
"""Default Trainer"""

import logging
import math
from statistics import mean

from tqdm import tqdm
import torch
import mlflow

from trainers.base_trainer import BaseTrainer
from models import get_model
from data import get_dataloader


log = logging.getLogger(__name__)


class DefaultTrainer(BaseTrainer):
    """DefaultTrainer
    
    Attributes:
        cfg: Config of project.
        model: Model.
    
    """

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Args:
            cfg: Config of project.
        """

        super().__init__(cfg)
        self.model = get_model(self.cfg)
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None


    def execute(self, eval: bool) -> None:
        """Execution
        Execute train or eval.
        Args:
            eval: For evaluation mode.
                True: Execute eval.
                False: Execute train.
        """

        if not eval:
            self.train_dataloader, self.val_dataloader = get_dataloader(self.cfg, mode="trainval")
            self.train()

        else:
            self.test_dataloader = get_dataloader(self.cfg, mode="test")
            self.eval()


    def train(self) -> None:
        """Train
        Trains model.
        """

        super().train()

        epochs = range(self.model.cfg.train.epochs)
        best_loss = math.inf

        with mlflow.start_run():
            self.log_params()

            for epoch in epochs:
                log.info(f"==================== Epoch: {epoch} ====================")
                log.info(f"Train:")
                self.model.network.train()
                losses = []

                with tqdm(self.train_dataloader, ncols=100) as pbar:
                    for idx, (inputs, targets) in enumerate(pbar):
                        inputs = inputs.to(self.model.device)
                        targets = targets.to(self.model.device)
                        outputs = self.model.network(inputs)

                        loss = self.model.criterion(outputs, targets)

                        loss.backward()

                        self.model.optimizer.step()
                        self.model.optimizer.zero_grad()

                        losses.append(loss.item())

                        pbar.set_description(f'train epoch:{epoch}')

                loss_avg = mean(losses)
                log.info(f"\tloss: {loss_avg}")
                metrics = {
                    "loss": loss_avg,
                }
                mlflow.log_metrics(metrics, step = epoch)

                if loss_avg < best_loss:
                    best_loss = loss_avg
                    self.model.save_ckpt(epoch=epoch, ckpt_path=self.cfg.train.ckpt_path)
                    log.info("Saved the check point.")

            log.info("Successfully trained the model.")

            self.log_artifacts()