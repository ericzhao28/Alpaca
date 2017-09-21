import os

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/"
EMBEDDING_SAVE = None
VOCAB_SAVE = None

BATCH_SIZE = None
NUMBERS = {
    'emb_dim': 300,
    'seq_len': 10,
    'gru_h': 20,
    'attention_h': 10,
    'state_h': 20,
    'n_entities': 20,
    'n_relationships': 20,

}

