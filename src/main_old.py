import os
import time
import ast

from src.features.feature_extractor import FeatureExtractor
from src.models.gmm_model import GMMModel
from src.models.svm_classifier import Classifier
from src.utils.utils import *
from src.utils.constants import *
from src.data.iam_dataset import IAMLoader


def load_data_set():
    processed_images = 0
    writers_features = {}
    test_writer_features = {}
    count = {}

    feature_extraction_elapsed_time = 0.0

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

            images_per_writer = 2

            # Get writer id.
            writer_id = writers_labels[filename]

            if writer_id not in count.keys():
                count[writer_id] = 0
            if writer_id not in writers_features.keys():
                writers_features[writer_id] = []
            if writer_id not in test_writer_features.keys():
                test_writer_features[writer_id] = []

            # Print image name.
            print('Processing form', filename, 'of writer', writer_id, '...')
            processed_images += 1

            # Read image in gray scale.
            org_img = cv.imread(IAMLoader.raw_data_form_path + filename, cv.IMREAD_GRAYSCALE)
            gray_img = cv.imread(IAMLoader.processed_data_form_gray_path + filename, cv.IMREAD_GRAYSCALE)
            bin_img = cv.imread(IAMLoader.processed_data_form_bin_path + filename, cv.IMREAD_GRAYSCALE)

            # Extract features.
            feature_extraction_start = time.clock()
            f = FeatureExtractor(org_img, gray_img, bin_img).extract()

            if count[writer_id] < images_per_writer:
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

    return writers_features, test_writer_features, processed_images, feature_extraction_elapsed_time


def evaluate_all_combinations_svm(writers_features, test_writer_features):
    # Try every possible combination of 2 writers.
    res_two = []
    processed = []
    output_file = open("two_writers_svm.txt", "w")
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
            f = []
            l = []
            f_t = []
            l_t = []
            for key, value in input_train_features.items():
                for i in value:
                    f.append(i)
                    l.append(key)

            for key, value in input_test_features.items():
                for i in value:
                    f_t.extend(i)
                    l_t.append(key)

            classifier = Classifier('svm', f, l, f_t, l_t)
            classifier.train()
            validation_accuracy = classifier.evaluate()

            res_two.append(validation_accuracy)
            output_file.write("Writers " + str(writer_a_id) + " and " + str(writer_b_id) + " Testing accuracy " + str(
                validation_accuracy) + '\n')
    output_file.close()

    # Try every possible combination of 3 writers.
    res_three = []
    processed = []
    output_file = open("three_writers_svm.txt", "w")
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

                f = []
                l = []
                f_t = []
                l_t = []
                for key, value in input_train_features.items():
                    for i in value:
                        f.append(i)
                        l.append(key)

                for key, value in input_test_features.items():
                    for i in value:
                        f_t.extend(i)
                        l_t.append(key)

                classifier = Classifier('svm', f, l, f_t, l_t)
                classifier.train()
                validation_accuracy = classifier.evaluate()

                res_three.append(validation_accuracy)

                output_file.write("Writers " + str(writer_a_id) + " and " + str(writer_b_id) + " and " + str(
                    writer_c_id) + " Testing accuracy " + str(validation_accuracy) + '\n')
    output_file.close()

    return np.mean(res_two), np.mean(res_three)


def evaluate_all_combinations_gmm(writers_features, test_writer_features):
    # Try every possible combination of 2 writers.
    res_two = []
    processed = []
    f = open("two_writers_gmm.txt", "w")
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
            res_two.append(validation_accuracy)
            f.write("Writers " + str(writer_a_id) + " and " + str(writer_b_id) + " Testing accuracy " + str(
                validation_accuracy) + '\n')
    f.close()

    # Try every possible combination of 3 writers.
    res_three = []
    processed = []
    f = open("three_writers_gmm.txt", "w")
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
                res_three.append(validation_accuracy)
                f.write("Writers " + str(writer_a_id) + " and " + str(writer_b_id) + " and " + str(
                    writer_c_id) + " Testing accuracy " + str(validation_accuracy) + '\n')
    f.close()
    return np.mean(res_two), np.mean(res_three)


if __name__ == '__main__':
    # Run the pre processor only once to get the preprocessed images (cropped ones).
    if GENERATE_PRE_PROCESSED_DATA:
        IAMLoader.generate_processed_data()

    # Load the data set
    writers_features, test_writers_features, processed_images_count, feature_extraction_elapsed_time = load_data_set()

    # Run predict on images.

    # Evaluate GMM model.
    (two_writers_gmm_acc, three_writers_gmm_acc) = evaluate_all_combinations_gmm(writers_features,
                                                                                 test_writers_features)

    # Evaluate SVM model.
    # (two_writers_svm_acc, three_writers_svm_acc) = evaluate_all_combinations_svm(writers_features,
    #                                                                              test_writers_features)
    # Print statistics.
    print("Processed ", processed_images_count, " images.")
    print("GMM 2 writers validation accuracy: ", two_writers_gmm_acc)
    print("GMM 3 writers validation accuracy: ", three_writers_gmm_acc)
    # print("SVM 2 writers validation accuracy: ", two_writers_svm_acc)
    # print("SVM 3 writers validation accuracy: ", three_writers_svm_acc)
    print("Feature extraction time: ", feature_extraction_elapsed_time)
