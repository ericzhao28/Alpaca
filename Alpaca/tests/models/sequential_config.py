import os


SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/sequential/"
GRAPHS_TRAIN_DIR = SAVES_DIR + "graphs/train/"
GRAPHS_TEST_DIR = SAVES_DIR + "graphs/test/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

ITERATIONS = 100
BATCH_SIZE = 17
N_FEATURES = 4
N_STEPS = 10
LAYERS = {
    'h_gru': 13,
    'h_att': 11,
}
ENCODED_DIM = 19
N_CLASSES = 10

