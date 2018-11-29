import os
import time

from src.features.feature_extractor import FeatureExtractor
from src.pre_processor import PreProcessor
from src.segmentation.line_segmentor import LineSegmentor
from src.utils.utils import *
from src.data.iam_dataset import get_writer_id

# Get start running time
start_time = time.time()

# Data set (i.e. lists of features and labels)
features = []
labels = []

# Data set path
data_path = "../data/raw/form"

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
        gray_img, bin_img = PreProcessor.process(gray_img, filename)

        # Line segment
        gray_lines, bin_lines = LineSegmentor.segment(gray_img, bin_img)

        # Extract features.
        features.append(FeatureExtractor.extract_features(gray_img, bin_img))

        # TODO Extract label (i.e. writer id).
        labels.append(get_writer_id(filename[:-4]))

        # Pass features and labels to a model for training.

    # Break in order not to enter other dirs in the data/raw/form folder
    break

# Get finish running time
finish_time = time.time()

# Print elapsed execution time
print("This took %.2f seconds" % (finish_time - start_time))
