import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from torchvision import datasets, transforms

from typing import Any, List, Tuple, Dict
from types import ModuleType

from sodium.utils import get_instance, setup_device, setup_param_groups, setup_logger, seed_everything

import sodium.model.loss as module_loss
import sodium.data_loader.augmentation as module_aug
import sodium.data_loader.data_loaders as module_data

from sodium.trainer import Trainer
import sodium.plot as plot

logger = setup_logger(__name__)


class Runner():
    def __init__(self, config):
        self.config = config

    def train(self, tsai_mode=False):
        cfg = self.config

        if tsai_mode:
            import sodium.tsai_model as module_arch
        else:
            import sodium.model.model as module_arch

        logger.info(f'Training: {cfg}')
        seed_everything(cfg['seed'])

        model = get_instance(module_arch, 'arch', cfg)

        model, device = setup_device(model, cfg['target_device'])

        param_groups = setup_param_groups(model, cfg['optimizer'])
        optimizer = get_instance(
            module_optimizer, 'optimizer', cfg, param_groups)

        transforms = get_instance(module_aug, 'augmentation', cfg)

        # get the train and test loaders
        train_loader, test_loader = get_instance(
            module_data, 'data_loader', cfg, transforms).get_loaders()

        if cfg['lr_scheduler']['type'] == 'OneCycleLR':
            logger.info('Building: torch.optim.lr_scheduler.OneCycleLR')
            sch_cfg = cfg['lr_scheduler']['args']
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=sch_cfg['max_lr'], steps_per_epoch=len(train_loader), epochs=cfg['training']['epochs'])
        else:
            lr_scheduler = get_instance(
                module_scheduler, 'lr_scheduler', cfg, optimizer)

        logger.info('Getting loss function handle')
        criterion = getattr(module_loss, cfg['criterion'])()

        logger.info('Initializing trainer')
        self.trainer = Trainer(model, criterion, optimizer, cfg, device,
                               train_loader, test_loader, lr_scheduler=lr_scheduler)

        self.trainer.train()

        logger.info('Finished!')

    def plot(self):
        logger.info('Plotting Metrics...')
        plot.plot_metrics(self.trainer.train_metric, self.trainer.test_metric)
