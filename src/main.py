import time
import random

from src.data.test_generator import TestGenerator
from src.pre_processing.pre_processor import PreProcessor
from src.features.feature_extractor import FeatureExtractor
from src.models.svm_classifier import Classifier
from src.utils.utils import *
from src.utils.constants import *


#
# Paths
#
data_path = '../data/data/'
wrong_data_path = '../data/wrong/'
results_path = '../data/results.txt'
time_path = '../data/time.txt'
expected_results_path = data_path + 'output.txt'

#
# Files
#
results_file = None
time_file = None

#
# Timers
#
total_time = 0.0
feature_extraction_time = 0.0


def run():
    # Set global variables
    global results_file, time_file, total_time

    # Start timer
    start = time.time()

    # Open results files
    results_file = open(results_path, 'w')
    time_file = open(time_path, 'w')

    # Iterate on every test iteration
    for root, dirs, files in os.walk(data_path):
        for d in dirs:
            print('Running test iteration', d, '...')

            # Start timer of test iteration
            t = time.time()

            #
            # Solve test iteration
            #
            try:
                r = process_test_iteration(data_path + d + '/')
            except:
                print('    >> Error')
                r = random.randint(1, 3)

            # Finish timer of test iteration
            t = (time.time() - t)

            # Write the results in files
            results_file.write(str(r) + '\n')
            time_file.write(format(t, '0.02f') + '\n')

            print('Finish test iteration %s in %.02f seconds' % (d, t))
            print()
        break

    # Close files
    results_file.close()
    time_file.close()

    # End timer
    total_time = (time.time() - start)


def process_test_iteration(path):
    features, labels, r = [], [], 0

    # Loop over every writer in the current test iteration
    # Should be 3 writers
    for root, dirs, files in os.walk(path):
        for d in dirs:
            print('    Processing writer', d, '...')
            x, y = process_writer(path + d + '/', int(d))
            features.extend(x)
            labels.extend(y)

    # Train the SVM model
    classifier = Classifier('svm', features, labels)
    classifier.train()

    # Loop over test images in the current test iteration
    # Should be 1 test image
    for root, dirs, files in os.walk(path):
        for filename in files:
            f = get_writing_features(path + filename)
            r = classifier.predict([f])[0]
            print('    Classifying test image \'%s\' as writer %d' % (path + filename, r))
            break
        break

    # Return classification
    return r


def process_writer(path, writer_id):
    x, y = [], []

    for root, dirs, files in os.walk(path):
        for filename in files:
            x.append(get_writing_features(path + filename))
            y.append(writer_id)

    return x, y


def get_writing_features(image_path):
    global feature_extraction_time
    org_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    gray_img, bin_img = PreProcessor.process(org_img)
    t = time.time()
    f = FeatureExtractor(org_img, gray_img, bin_img).extract()
    t = (time.time() - t)
    feature_extraction_time += t
    return f


def calculate_accuracy():
    # Read results
    with open(results_path) as f:
        predicted_res = f.read().splitlines()
    with open(expected_results_path) as f:
        expected_res = f.read().splitlines()

    # Calculate accuracy
    cnt = 0
    for i in range(len(predicted_res)):
        if predicted_res[i] == expected_res[i]:
            cnt += 1

    # Return accuracy
    return cnt, len(predicted_res)


def save_wrong_iterations():
    # Read results
    with open(results_path) as f:
        predicted_res = f.read().splitlines()
    with open(expected_results_path) as f:
        expected_res = f.read().splitlines()

    # Create wrong classified data directory
    if not os.path.exists(wrong_data_path):
        os.makedirs(wrong_data_path)

    # Open expected output text file
    file = open(wrong_data_path + 'output.txt', 'a+')

    # Get number of previously wrong classified iterations
    k = 0
    for root, dirs, files in os.walk(wrong_data_path):
        k = len(dirs)
        break

    # Save wrong classifications
    for i in range(len(predicted_res)):
        if predicted_res[i] == expected_res[i]:
            continue

        # Copy
        src_path = data_path + format(i, '03d') + '/'
        dst_path = wrong_data_path + format(k, '03d') + '/'
        shutil.copytree(src_path, dst_path)
        k += 1

        # Write expected result
        file.write(expected_res[i] + '\n')

        # Log
        print('Wrong classification in iteration: %s (output: %s, expected: %s)' %
              (format(i, '03d'), predicted_res[i], expected_res[i]))

    # Close file
    file.close()


#
# Generate test iterations from IAM data set
#
if GENERATE_TEST_ITERATIONS:
    gen = TestGenerator()
    gen.generate(data_path, 50, 3, 2)

#
# Main
#
run()
print('-------------------------------')
print('Total elapsed time: %.2f seconds' % total_time)
print('Feature extraction elapsed time: %.2f seconds' % feature_extraction_time)
# TODO: to be removed before discussion
print('Classification accuracy: %d/%d' % calculate_accuracy())
print('-------------------------------')
print()
# TODO: to be removed before discussion
save_wrong_iterations()

