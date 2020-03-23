import yaml

from typing import Any, List, Tuple, Dict
from types import ModuleType

import torch
import torch.nn as nn

from . import setup_logger

logger = setup_logger(__name__)


def load_config(filename: str) -> dict:
    """
    Load a configuration file as YAML
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    return config


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
