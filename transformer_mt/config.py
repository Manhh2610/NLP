# ===== Dataset =====
SRC_LANG = "en"
TGT_LANG = "de"
MIN_FREQ = 2
BATCH_SIZE = 32
MAX_LEN = 100

# ===== Model =====
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 3
D_FF = 512
DROPOUT = 0.1

# ===== Training =====
N_EPOCHS = 20
LR = 1e-4
CLIP = 1.0

# ===== Device =====
DEVICE = "cuda"  
# ===== Special tokens =====
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
