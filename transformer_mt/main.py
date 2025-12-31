import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloader
from model.encoder import Encoder
from model.decoder import Decoder
from model.transformer import Transformer
from train import train_epoch, evaluate, calculate_bleu


def main():
    device = torch.device(config.DEVICE)

    # ===== Load data =====
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloader(batch_size=config.BATCH_SIZE)

    PAD_IDX = tgt_vocab[config.PAD_TOKEN]

    # ===== Build model =====
    encoder = Encoder(
        vocab_size=len(src_vocab),
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
    )

    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
    )

    model = Transformer(encoder, decoder, PAD_IDX, device).to(device)

    # ===== Optimizer & Loss =====
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_valid_loss = float("inf")

    # ===== Training loop =====
    for epoch in range(config.N_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, config.CLIP
        )
        valid_loss = evaluate(
            model, valid_loader, criterion
        )

        print(f"Epoch {epoch+1:02}")
        print(f"  Train loss: {train_loss:.3f}")
        print(f"  Valid loss: {valid_loss:.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_transformer.pt")

    # ===== BLEU =====
    model.load_state_dict(torch.load("best_transformer.pt"))
    bleu = calculate_bleu(model, test_loader, tgt_vocab, device)
    print(f"BLEU score: {bleu:.4f}")


if __name__ == "__main__":
    main()
