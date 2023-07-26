import math
import torch
from torch import nn, Tensor


class PositionEncodingSine1D(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self,
                 d_model: int,
                 max_len: int = 4096):
        """
        Args:
            d_model (int):
            max_len (int): the max length of 4096
        """
        super().__init__()

        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros((max_len, 1, d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe, persistent=False)  # [1, C, H, W]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]
