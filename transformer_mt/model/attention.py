import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [seq_len, batch, d_model]
        """

        seq_len, batch, _ = Q.shape

        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = Q.view(seq_len, batch, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(seq_len, batch, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(seq_len, batch, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(seq_len, batch, self.d_model)

        return self.fc(output)
