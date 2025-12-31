import math
import torch
import torch.nn as nn


def subsequent_mask(size):
    """
    Mask để decoder không nhìn token tương lai
    shape: [1, size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        return x + self.pe[:x.size(0)].unsqueeze(1)
