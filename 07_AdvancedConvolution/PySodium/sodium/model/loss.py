import torch.nn.functional as F


def nll_loss(output, target, **kwargs):
    return F.nll_loss(output, target, **kwargs)
