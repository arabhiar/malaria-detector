# importing packages
import os

# initializing the path where the downloaded and extracted data is present
DOWNLOADED_DATASET = "malaria/cell_images"

# this is the base path for our final dataset
BASE_PATH = "malaria"

TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
