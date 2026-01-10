import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_PATHS = {
    "mit-states": os.path.join(DIR_PATH, "/root/autodl-tmp/data/CZSL/mit-states"),
    "ut-zappos": os.path.join(DIR_PATH, "/root/autodl-tmp/data/CZSL/ut-zappos"),
    "cgqa": os.path.join(DIR_PATH, "/root/autodl-tmp/data/CZSL/cgqa"),
    "multi-attrs":os.path.join(DIR_PATH, "/root/autodl-tmp/data/CZSL/multi_attr")
}