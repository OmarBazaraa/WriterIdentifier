import os
import time

from src.features.feature_extractor import FeatureExtractor
from src.pre_processor import PreProcessor
from src.utils.utils import *
# from src.data.iam_dataset import get_writer_id

# Get start running time
start_time = time.clock()

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
        img = cv.imread(data_path + "/" + filename, cv.IMREAD_GRAYSCALE)

        # Pre process image.
        gray_img, bin_img = PreProcessor.process(img, filename)

        # Extract features.
        extractor = FeatureExtractor(img, gray_img, bin_img)
        f = extractor.extract()
        features.append(f)

        print(f)

        # Extract label (i.e. writer id).
        # writer_id = get_writer_id(filename[:-4])
        # labels.append(writer_id)
        break

    # Break in order not to enter other dirs in the data/raw/form folder
    break

# Pass features and labels to a model for training.
# classifier = SVMClassifier()
# classifier.train(features, labels)

# TODO Predict a feature vector.

# Get finish running time
finish_time = time.clock()

# Print elapsed execution time
print("This took %.2f seconds" % (finish_time - start_time))
