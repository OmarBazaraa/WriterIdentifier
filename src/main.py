import os
import time
import ast

from src.features.feature_extractor import FeatureExtractor
from src.models.train_model import SVMClassifier
from src.utils.utils import *
from src.data.iam_dataset import IAMLoader

# Get start running time.
start_time = time.clock()
feature_extraction_elapsed_time = 0.0
training_elapsed_time = 0.0

# Run the pre processor only once to get the preprocessed images (cropped ones).
if GENERATE_PRE_PROCESSED_DATA:
    IAMLoader.generate_processed_data()

# Data set (i.e. lists of features and labels).
features = []
labels = []

# Read labels.
s = open(IAMLoader.processed_data_writer_ids, 'r').read()
labels = ast.literal_eval(s)

# Walk on data set directory
for root, dirs, files in os.walk(IAMLoader.processed_data_images_path + "/gray/"):
    #
    # Loop on every file in the directory.
    #
    for filename in files:
        # Ignore gitignore file.
        if filename[0] == '.' or filename in IAMLoader.images_of_interest:
            continue

        # Print image name.
        print(filename)

        # Read image in gray scale.
        org_img = cv.imread(IAMLoader.raw_data_path + "/" + filename, cv.IMREAD_GRAYSCALE)
        gray_img = cv.imread(IAMLoader.processed_data_images_path + "/gray/" + filename, cv.IMREAD_GRAYSCALE)
        bin_img = cv.imread(IAMLoader.processed_data_images_path + "/bin/" + filename, cv.IMREAD_GRAYSCALE)

        # Extract features.
        feature_extraction_start = time.clock()
        extractor = FeatureExtractor(org_img, gray_img, bin_img)
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
