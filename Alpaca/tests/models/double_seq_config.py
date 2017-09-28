import os


SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/double_seq/"
GRAPHS_TRAIN_DIR = SAVES_DIR + "graphs/train/"
GRAPHS_TEST_DIR = SAVES_DIR + "graphs/test/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

ITERATIONS = 100
BATCH_SIZE = 17
N_FEATURES = 4
N_SEQS = 15
N_STEPS = 10
LAYERS = {
    'h_step_gru': 11,
    'h_step_att': 7,
    'h_seq_gru': 13,
    'h_seq_att': 9,
}
SEQ_ENCODED_DIM = 19
ENCODED_DIM = 23
N_CLASSES = 10

