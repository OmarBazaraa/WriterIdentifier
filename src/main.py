import os
import time
import ast

from src.features.feature_extractor import FeatureExtractor
from src.models.gmm_model import GMMModel
from src.utils.utils import *
from src.utils.constants import *
from src.data.iam_dataset import IAMLoader

# Set timers.
start_time = time.clock()
feature_extraction_elapsed_time = 0.0
training_elapsed_time = 0.0

# Data set (i.e. lists of features and labels).
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
        # if writer_id not in IAMLoader.top_writers_ids:
        #     continue

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

# Filter writers who don't have 2 images in train or no image in the test.
tmp_writers_features = writers_features.copy()
for writer_id, features in tmp_writers_features.items():
    # Remove from the train and the test
    if count[writer_id] < 1 or len(test_writer_features[writer_id]) == 0:
        del writers_features[writer_id]
        del test_writer_features[writer_id]

print("number of writers ", len(writers_features.keys()))
# # Try every possible combination of 2 writers.
res = []
processed = []
f = open("two_writers.txt", "w")
for writer_a_id, a_features in writers_features.items():
    for writer_b_id, b_features in writers_features.items():
        if writer_a_id == writer_b_id:
            continue
        if sorted((writer_a_id, writer_b_id)) in processed:
            continue
        processed.append(sorted((writer_a_id, writer_b_id)))
        input_train_features = {writer_a_id: a_features, writer_b_id: b_features}
        input_test_features = {writer_a_id: test_writer_features[writer_a_id],
                               writer_b_id: test_writer_features[writer_b_id]}
        gmm_model = GMMModel(input_train_features)
        gmm_model.get_writers_models()
        validation_accuracy = gmm_model.evaluate(input_test_features)
        res.append(validation_accuracy)
        f.write("Writers " + str(writer_a_id) + " and " + str(writer_b_id) + " Testing accuracy " + str(
            validation_accuracy) + '\n')
f.close()

# Try every possible combination of 3 writers.
res = []
processed = []
f = open("three_writers.txt", "w")
for writer_a_id, a_features in writers_features.items():
    for writer_b_id, b_features in writers_features.items():
        if writer_a_id == writer_b_id:
            continue
        for writer_c_id, c_features in writers_features.items():
            if writer_c_id == writer_a_id or writer_c_id == writer_b_id:
                continue

            if sorted((writer_a_id, writer_b_id, writer_c_id)) in processed:
                continue

            processed.append(sorted((writer_a_id, writer_b_id, writer_c_id)))

            input_train_features = {writer_a_id: a_features, writer_b_id: b_features, writer_c_id: c_features}
            input_test_features = {writer_a_id: test_writer_features[writer_a_id],
                                   writer_b_id: test_writer_features[writer_b_id],
                                   writer_c_id: test_writer_features[writer_c_id]}
            gmm_model = GMMModel(input_train_features)
            gmm_model.get_writers_models()
            validation_accuracy = gmm_model.evaluate(input_test_features)
            res.append(validation_accuracy)
            f.write("Writers " + str(writer_a_id) + " and " + str(writer_b_id) + " and " + str(
                writer_c_id) + " Testing accuracy " + str(validation_accuracy) + '\n')
f.close()
# training_elapsed_time = time.clock()

# Train and evaluate the classifier using its test set.
# SVM Classifier.
# f = []
# l = []
# f_t = []
# l_t = []
# for key, value in writers_features.items():
#     for i in value:
#         f.append(i)
#         l.append(key)
#
# for key, value in test_writer_features.items():
#     for i in value:
#         f_t.extend(i)
#         l_t.append(key)
#
# classifier = Classifier('svm', f, l, f_t, l_t)
# classifier.train()
# training_elapsed_time = (time.clock() - training_elapsed_time)
# validation_accuracy = classifier.evaluate()

# Get finish running time.
finish_time = time.clock()

#
# Print statistics.
#
# print("Processed ", len(labels), " images.")
# print("Train Accuracy rate: ", training_accuracy, '%')
# print("Validation Accuracy rate: ", validation_accuracy, '%')
# print("Feature extraction elapsed time: %.2f seconds" % feature_extraction_elapsed_time)
# print("Training elapsed time: %.2f seconds" % training_elapsed_time)
# print("Total elapsed time: %.2f seconds" % (finish_time - start_time))
