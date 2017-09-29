import os


SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/"
GRAPHS_TRAIN_DIR = SAVES_DIR + "graphs/train/"
GRAPHS_TEST_DIR = SAVES_DIR + "graphs/test/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

ITERATIONS = 32
BATCH_SIZE = 20
EMB_DIM = 300
EMB_PATH = os.path.dirname(os.path.realpath(__file__)) + "/glove.vector"
N_STEPS = 10
LAYERS = {
    'h_seq_int': 11,
    'h_seq_att': 19,
    'h_seq_output': 7,
    'h_state_int': 13,
    'h_state_output': 17,
}
N_CLASSES = 200

