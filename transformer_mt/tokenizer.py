# tokenizer.py
import io
import spacy
from collections import Counter
from tokenizers import Tokenizer

from config import *

en_tok = spacy.load("en_core_web_sm")
de_tok = spacy.load("de_core_news_sm")

def build_vocab(path, tokenizer):
    counter = Counter()
    with io.open(path, encoding="utf8") as f:
        for line in f:
            counter.update([tok.text.lower() for tok in tokenizer(line)])

    v = vocab(
        counter,
        specials=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN],
        min_freq=MIN_FREQ
    )
    v.set_default_index(v[UNK_TOKEN])
    return v
