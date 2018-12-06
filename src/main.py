import os
import time
import ast


from src.features.feature_extractor import FeatureExtractor
from src.models.train_model import *
from src.utils.utils import *
from src.utils.constants import *
from src.data.iam_dataset import IAMLoader


# Set timers.
start_time = time.clock()
feature_extraction_elapsed_time = 0.0
training_elapsed_time = 0.0

# Data set (i.e. lists of features and labels).
features = []
labels = []
writers_features = {}
test_writer_features = {}
count = {}

# Run the pre processor only once to get the preprocessed images (cropped ones).
if GENERATE_PRE_PROCESSED_DATA:
    IAMLoader.generate_processed_data()

# Read writers IDs.
s = open(IAMLoader.processed_data_writer_ids, 'r').read()
writers_labels = ast.literal_eval(s)

# Walk on data set directory.
for root, dirs, files in os.walk(IAMLoader.processed_data_form_gray_path):
    #
    # Loop on every file in the directory.
    #
    for filename in files:
        # Ignore git ignore file.
        if filename[0] == '.':
            continue

        # TODO: to be fixed and removed.
        if filename in IAMLoader.images_of_interest:
            continue

        IMAGES_PER_WRITER = 2

        # Get writer id.
        writer_id = writers_labels[filename]

        # TODO: just for testing.
        if writer_id not in IAMLoader.top_writers_ids:
            continue

        if writer_id not in count.keys():
            count[writer_id] = 0
        if writer_id not in writers_features.keys():
            writers_features[writer_id] = []
        if writer_id not in test_writer_features.keys():
            test_writer_features[writer_id] = []

        # if len(test_writer_features[writer_id]) > 0 and count[writer_id] >= IMAGES_PER_WRITER:
        #     continue

        # Print image name.
        print('Processing form', filename, 'of writer', writer_id, '...')

        # Append labels
        labels.append(writer_id)

        # Read image in gray scale.
        org_img = cv.imread(IAMLoader.raw_data_form_path + filename, cv.IMREAD_GRAYSCALE)
        gray_img = cv.imread(IAMLoader.processed_data_form_gray_path + filename, cv.IMREAD_GRAYSCALE)
        bin_img = cv.imread(IAMLoader.processed_data_form_bin_path + filename, cv.IMREAD_GRAYSCALE)

        #
        # Extract features.
        #
        feature_extraction_start = time.clock()
        f = FeatureExtractor(org_img, gray_img, bin_img).extract()

        if count[writer_id] < IMAGES_PER_WRITER:
            writers_features[writer_id].extend(f)
            count[writer_id] += 1
        else:
            test_writer_features[writer_id].append(f)

        feature_extraction_elapsed_time += (time.clock() - feature_extraction_start)

    # Break in order not to enter other dirs in the data/raw/form folder.
    break

#
# Train and evaluate the classifier using its test set.
#
training_elapsed_time = time.clock()
gmm_model = GMMModel(writers_features)
gmm_model.get_writers_models()
training_elapsed_time = (time.clock() - training_elapsed_time)
(training_accuracy, validation_accuracy) = gmm_model.evaluate(test_writer_features)

#
# Pass features and labels to a model for training.
#
# training_elapsed_time = time.clock()
# classifier = Classifier('mlp', features, labels)
# classifier.train()
# training_elapsed_time = (time.clock() - training_elapsed_time)

# Get finish running time.
finish_time = time.clock()

#
# Print statistics.
#
print("Processed ", len(labels), " images.")
print("Train Accuracy rate: ", training_accuracy, '%')
print("Validation Accuracy rate: ", validation_accuracy, '%')
print("Feature extraction elapsed time: %.2f seconds" % feature_extraction_elapsed_time)
print("Training elapsed time: %.2f seconds" % training_elapsed_time)
print("Total elapsed time: %.2f seconds" % (finish_time - start_time))
