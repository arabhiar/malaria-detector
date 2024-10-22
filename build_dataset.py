from imutils import paths
import random
import shutil
import os
from model_design import config

imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(69)
random.shuffle(imagePaths)

i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VAL_PATH),
    ("testing", testPaths, config.TEST_PATH),
]

for (dType, imagePaths, baseOutput) in datasets:
    print("[INFO] 'building {}' split".format(dType))
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)
    for inputPath in imagePaths:
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]
        labelPath = os.path.sep.join([baseOutput, label])
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)
