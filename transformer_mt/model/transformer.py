import torch
import torch.nn as nn
from utils import subsequent_mask


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        src: [seq_len, batch]
        """
        return (src != self.pad_idx).permute(1, 0).unsqueeze(1)

    def make_tgt_mask(self, tgt):
        """
        tgt: [seq_len, batch]
        """
        tgt_pad_mask = (tgt != self.pad_idx).permute(1, 0).unsqueeze(1)
        seq_len = tgt.size(0)
        tgt_sub_mask = subsequent_mask(seq_len).to(self.device)
        return tgt_pad_mask & (~tgt_sub_mask)

    def forward(self, src, tgt):
        """
        src: [src_len, batch]
        tgt: [tgt_len, batch]
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_out, tgt_mask, src_mask)

        return output
