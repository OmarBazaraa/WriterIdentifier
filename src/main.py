import os
import time

from src.pre_processor import PreProcessor
from src.features.feature_extractor import FeatureExtractor
from src.models.train_model import SVMClassifier
from src.utils.utils import *
from src.data.iam_dataset import get_writer_id

# Get start running time.
start_time = time.clock()
feature_extraction_elapsed_time = 0.0
training_elapsed_time = 0.0

# Failed images.
images_of_interest = ['h07-084.png', 'h07-080a.png', 'h07-075a.png', 'h07-078a.png', 'k03-164.png', 'k02-053.png', 'j07-005.png']

# Data set (i.e. lists of features and labels).
features = []
labels = []

# Data set path
data_path = "../data/raw/form"

# Walk on data set directory
for root, dirs, files in os.walk(data_path + "/"):
    #
    # Loop on every file in the directory.
    #
    for filename in files:
        # Ignore gitignore file.
        if filename[0] == '.' or filename in images_of_interest:
            continue

        # Extract label (i.e. writer id).
        writer_id = get_writer_id(filename[:-4])
        labels.append(writer_id)

        # Print image name.
        print(filename)

        # Read image in gray scale.
        img = cv.imread(data_path + "/" + filename, cv.IMREAD_GRAYSCALE)

        # Pre process image.
        gray_img, bin_img = PreProcessor.process(img, filename)

        if gray_img == [] or bin_img == []:
            continue

        # Extract features.
        feature_extraction_start = time.clock()
        extractor = FeatureExtractor(img, gray_img, bin_img)
        f = extractor.extract()
        features.append(f)
        feature_extraction_elapsed_time += (time.clock() - feature_extraction_start)
    # Break in order not to enter other dirs in the data/raw/form folder.
    break

# Pass features and labels to a model for training.
start_training_time = 0.0
classifier = SVMClassifier(features, labels)
classifier.train()
end_training_time = time.clock()

# Evaluate the classifier using its test set.
validation_accuracy = classifier.evaluate()

# TODO Predict a label using a feature vector.

# Get finish running time.
finish_time = time.clock()

# Print statistics.
print("Processed ", len(labels), " images.")
print("Accuracy rate: ", validation_accuracy, '%')
print("Feature extraction elapsed time: %.2f seconds" % feature_extraction_elapsed_time)
print("Training elapsed time: %.2f seconds" % (end_training_time - start_training_time))
print("This took %.2f seconds" % (finish_time - start_time))
