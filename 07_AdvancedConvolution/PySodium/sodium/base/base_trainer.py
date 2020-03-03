from pathlib import Path

import yaml
import torch

from sodium.utils import setup_logger

logger = setup_logger(__name__)


class BaseTrainer:
    """Base Trainer for all models
    """

    def __init__(self, model, loss, optimizer, config, device):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self._setup_monitoring(config['training'])

    def train(self):
        logger.info('Starting training ...')
        for epoch in range(1, self.epochs):
            print(f'EPOCH: {epoch}')

            self._train_epoch(epoch)

    def _train_epoch(self, epoch: int) -> dict:
        raise NotImplementedError

    def _setup_monitoring(self, config: dict) -> None:
        self.epochs = config['epochs']
