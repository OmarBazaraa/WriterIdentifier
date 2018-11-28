import os
import cv2 as cv
import numpy as np
import time

from src.pre_processor import PreProcessor
from src.segmentation.line_segmentor import LineSegmentor
from src.feature_extractor import FeatureExtractor
from src.utils.utils import *

# Get start running time
start_time = time.time()

# Data set (i.e. lists of features and labels)
features = []
labels = []

# Data set path
data_path = "../data/"

# Walk on data set directory
for root, dirs, files in os.walk(data_path + "/"):
    #
    # Loop on every file in the directory
    #
    for filename in files:
        # Ignore gitignore file.
        if filename[0] == '.':
            continue

        # Print image name
        print(filename)

        # Read image in gray scale.
        gray_img = cv.imread(data_path + "/" + filename, cv.IMREAD_GRAYSCALE)

        # Pre process image.
        gray_img, bin_img = PreProcessor.process(gray_img)

        # Line segment
        gray_lines, bin_lines = LineSegmentor.segment(gray_img, bin_img)

        # Extract features.
        features.append(FeatureExtractor.extract_features(gray_img, bin_img))

        # TODO Extract label (i.e. writer id).

# Get finish running time
finish_time = time.time()

# Print elapsed execution time
print("This took %.2f seconds" % (finish_time - start_time))
