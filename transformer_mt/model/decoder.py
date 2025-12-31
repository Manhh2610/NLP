import torch.nn as nn
from model.attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_attn = MultiHeadAttention(d_model, n_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        attn2 = self.enc_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        """
        tgt: [seq_len, batch]
        """
        x = self.embed(tgt)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)

        return self.fc_out(x)
