'''
Standard configuration for apple support.
'''

import os


DUMPS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

RAW_SAVE_NAME = "raw_dataset.json"
PROCESSED_ENT_SAVE_NAME = "processed_ent_dataset.p"
PROCESSED_REL_SAVE_NAME = "processed_rel_dataset.p"

