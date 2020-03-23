import torch.nn.functional as F
import torch.nn as nn


def nll_loss():
    return F.nll_loss


def cross_entropy_loss():
    return nn.CrossEntropyLoss()
