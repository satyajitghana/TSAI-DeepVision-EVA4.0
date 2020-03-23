import torch.nn as nn
from torchsummary import summary


class BaseModel(nn.Module):
    """Base Class for all models
    """

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        """Forward pass logic

        Returns
            Model output
        """
        raise NotImplementedError
