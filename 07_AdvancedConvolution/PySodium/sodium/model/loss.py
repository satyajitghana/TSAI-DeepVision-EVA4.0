import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target, **kwargs):
    return F.nll_loss(output, target, **kwargs)


def cross_entropy_loss(output, target, **kwargs):
    return nn.CrossEntropyLoss(output, target)
