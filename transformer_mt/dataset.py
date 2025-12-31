import io
import torch
import spacy
from collections import Counter
from torch.utils.data import DataLoader
from datasets import load_dataset
dataset = load_dataset("multi30k", "de-en")

import config

# ===== Load tokenizer =====
en_tok = spacy.load("en_core_web_sm")
de_tok = spacy.load("de_core_news_sm")


def tokenize_en(text):
    return [tok.text.lower() for tok in en_tok(text)]


def tokenize_de(text):
    return [tok.text.lower() for tok in de_tok(text)]


# ===== Download Multi30K =====
URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
FILES = {
    "train": ("train.en.gz", "train.de.gz"),
    "valid": ("val.en.gz", "val.de.gz"),
    "test":  ("test_2016_flickr.en.gz", "test_2016_flickr.de.gz"),
}


def load_raw_files(split):
    en_file, de_file = FILES[split]
    en_path = extract_archive(download_from_url(URL + en_file))[0]
    de_path = extract_archive(download_from_url(URL + de_file))[0]
    return en_path, de_path


# ===== Build vocabulary =====
def build_vocab(path, tokenizer):
    counter = Counter()
    with io.open(path, encoding="utf8") as f:
        for line in f:
            counter.update(tokenizer(line))

    v = vocab(
        counter,
        specials=[
            config.UNK_TOKEN,
            config.PAD_TOKEN,
            config.BOS_TOKEN,
            config.EOS_TOKEN,
        ],
        min_freq=config.MIN_FREQ,
    )
    v.set_default_index(v[config.UNK_TOKEN])
    return v


# ===== Convert text → tensor =====
def data_process(paths, src_vocab, tgt_vocab):
    data = []
    with io.open(paths[0], encoding="utf8") as src_f, \
         io.open(paths[1], encoding="utf8") as tgt_f:

        for src, tgt in zip(src_f, tgt_f):
            src_ids = torch.tensor(
                [src_vocab[t] for t in tokenize_en(src)]
            )
            tgt_ids = torch.tensor(
                [tgt_vocab[t] for t in tokenize_de(tgt)]
            )
            data.append((src_ids, tgt_ids))

    return data


# ===== Collate function =====
def collate_fn(batch, pad_idx, bos_idx, eos_idx, device):
    src_batch, tgt_batch = [], []

    for src, tgt in batch:
        src_batch.append(
            torch.cat([torch.tensor([bos_idx]), src, torch.tensor([eos_idx])])
        )
        tgt_batch.append(
            torch.cat([torch.tensor([bos_idx]), tgt, torch.tensor([eos_idx])])
        )

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, padding_value=pad_idx
    )
    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, padding_value=pad_idx
    )

    return src_batch.to(device), tgt_batch.to(device)


# ===== Public API =====
def get_dataloader(batch_size):
    device = torch.device(config.DEVICE)

    # Load raw paths
    train_paths = load_raw_files("train")
    valid_paths = load_raw_files("valid")
    test_paths  = load_raw_files("test")

    # Build vocab from training set
    src_vocab = build_vocab(train_paths[0], tokenize_en)
    tgt_vocab = build_vocab(train_paths[1], tokenize_de)

    PAD_IDX = src_vocab[config.PAD_TOKEN]
    BOS_IDX = src_vocab[config.BOS_TOKEN]
    EOS_IDX = src_vocab[config.EOS_TOKEN]

    # Process data
    train_data = data_process(train_paths, src_vocab, tgt_vocab)
    valid_data = data_process(valid_paths, src_vocab, tgt_vocab)
    test_data  = data_process(test_paths,  src_vocab, tgt_vocab)

    # DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b, PAD_IDX, BOS_IDX, EOS_IDX, device
        ),
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b, PAD_IDX, BOS_IDX, EOS_IDX, device
        ),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(
            b, PAD_IDX, BOS_IDX, EOS_IDX, device
        ),
    )

    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab
