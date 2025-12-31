import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu


def train_epoch(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for src, tgt in tqdm(loader):
        optimizer.zero_grad()

        output = model(src, tgt[:-1])

        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        tgt = tgt[1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in loader:
            output = model(src, tgt[:-1])

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            tgt = tgt[1:].reshape(-1)

            loss = criterion(output, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def calculate_bleu(model, loader, tgt_vocab, device):
    model.eval()
    refs, hyps = [], []

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            output = model(src, tgt[:-1])
            pred = output.argmax(-1)

            for i in range(pred.shape[1]):
                hyp = []
                for tok in pred[:, i]:
                    if tok.item() == tgt_vocab["<eos>"]:
                        break
                    hyp.append(tok.item())

                ref = tgt[:, i].tolist()
                refs.append([ref])
                hyps.append(hyp)

    return corpus_bleu(refs, hyps)
