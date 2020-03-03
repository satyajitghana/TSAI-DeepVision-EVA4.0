import argparse
import os
import random
from typing import Any, List, Tuple, Dict
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from torchvision import datasets, transforms

from sodium.utils import setup_logger, load_config, seed_everything
from sodium.trainer import Trainer

import sodium.model.model as module_arch
import sodium.model.loss as module_loss
import sodium.data_loader.augmentation as module_aug
import sodium.data_loader.data_loaders as module_data

logger = setup_logger(__name__)


def train(cfg: Dict) -> None:
    logger.info(f'Training: {cfg}')
    seed_everything(cfg['seed'])

    model = get_instance(module_arch, 'arch', cfg)

    model, device = setup_device(model, cfg['target_device'])

    param_groups = setup_param_groups(model, cfg['optimizer'])
    optimizer = get_instance(module_optimizer, 'optimizer', cfg, param_groups)
    lr_scheduler = get_instance(
        module_scheduler, 'lr_scheduler', cfg, optimizer)

    transforms = get_instance(module_aug, 'augmentation', cfg)
    train_loader = get_instance(module_data, 'data_loader', cfg, transforms)
    test_loader = train_loader.test_split()

    logger.info('Getting loss function handle')
    loss = getattr(module_loss, cfg['loss'])

    logger.info('Initializing trainer')
    trainer = Trainer(model, loss, optimizer, cfg, device,
                      train_loader, test_loader, lr_scheduler=lr_scheduler)

    trainer.train()

    logger.info('Finished!')


def setup_device(model: nn.Module, target_device: int) -> Tuple[torch.device, List[int]]:
    available_devices = list(range(torch.cuda.device_count()))
    logger.info(
        f'Using device {target_device} of available devices {available_devices}')

    device = torch.device(f'cuda:{target_device}')
    model = model.to(device)

    return model, device


def setup_param_groups(model: nn.Module, config: Dict) -> List:
    return [{'params': model.parameters(), **config}]


def get_instance(module: ModuleType, name: str, config: Dict, *args: Any) -> Any:
    ctor_name = config[name]['type']
    logger.info(f'Building: {module.__name__}.{ctor_name}')
    return getattr(module, ctor_name)(*args, **config[name]['args'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Sodium Model')

    parser.add_argument('-c', '--config', default=None,
                        type=str, help='config file path (default: None)')

    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    # parse the arguments
    args = parser.parse_args()

    config = load_config(args.config)

    train(config)
