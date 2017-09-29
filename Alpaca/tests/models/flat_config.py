import os


SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/flat/"
GRAPHS_TRAIN_DIR = SAVES_DIR + "graphs/train/"
GRAPHS_TEST_DIR = SAVES_DIR + "graphs/test/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

ITERATIONS = 200
BATCH_SIZE = 19
N_FEATURES = 4
N_CLASSES = 10
LAYERS = {
    'h_one_encoder': 17,
    'h_one_encoded': 13,
    'h_two_encoder': 11,
    'h_two_encoded': 7
}

