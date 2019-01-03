import time
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.data.test_generator import TestGenerator
from src.pre_processing.pre_processor import PreProcessor
from src.segmentation.line_segmentor import LineSegmentor
from src.features.feature_extractor import FeatureExtractor
from src.utils.utils import *
from src.utils.constants import *


#
# Files
#
results_file = None
time_file = None

#
# Timers
#
total_time = 0.0

# =====================================================================
#
# Pipeline functions
#
# =====================================================================


def run():
    # Set global variables
    global results_file, time_file, total_time

    # Start timer
    start = time.time()

    # Open results files
    results_file = open(RESULTS_PATH, 'w')
    time_file = open(ELAPSED_TIME_PATH, 'w')

    # Iterate on every test case
    for root, dirs, files in os.walk(TEST_CASES_PATH):
        for d in dirs:
            print('Running test iteration', d, '...')

            # Start timer of test iteration
            t = time.time()

            # Solve test iteration
            try:
                r = process_test_case(TEST_CASES_PATH + d + '/')
            except:
                print('    >> Error')
                r = random.randint(1, 3)

            # Finish timer of test iteration
            t = (time.time() - t)

            # Write the results in files
            results_file.write(str(r) + '\n')
            time_file.write(format(t, '0.02f') + '\n')

            print('Finish test iteration %s in %.02f seconds\n' % (d, t))
        break

    # Close files
    results_file.close()
    time_file.close()

    # End timer
    total_time = (time.time() - start)


def process_test_case(path):
    features, labels, r = [], [], 0

    # Loop over every writer in the current test iteration
    # Should be 3 writers
    for root, dirs, files in os.walk(path):
        for d in dirs:
            print('    Processing writer', d, '...')
            x, y = get_writer_features(path + d + '/', d)
            features.extend(x)
            labels.extend(y)

    # Train the SVM model
    classifier = SVC(C=5.0, gamma='auto', probability=True)
    classifier.fit(features, labels)

    # Loop over test images in the current test iteration
    # Should be 1 test image
    for root, dirs, files in os.walk(path):
        for filename in files:
            # Extract the features of the text image
            x = get_features(path + filename)

            # Get the most likely writer
            p = classifier.predict_proba(x)
            p = np.sum(p, axis=0)
            r = classifier.classes_[np.argmax(p)]

            print('    Classifying test image \'%s\' as writer \'%s\'' % (path + filename, r))
        break

    # Return classification
    return r


def get_writer_features(path, writer_id):
    # All lines of the writer
    total_gray_lines, total_bin_lines = [], []

    # Read and append all lines of the writer
    for root, dirs, files in os.walk(path):
        for filename in files:
            gray_img = cv.imread(path + filename, cv.IMREAD_GRAYSCALE)
            gray_img, bin_img = PreProcessor.process(gray_img)
            gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()
            total_gray_lines.extend(gray_lines)
            total_bin_lines.extend(bin_lines)
        break

    # Extract features of every line separately
    x, y = [], []
    for g, b in zip(total_gray_lines, total_bin_lines):
        f = FeatureExtractor([g], [b]).extract()
        x.append(f)
        y.append(writer_id)

    return x, y


def get_features(path):
    # Read and pre-process the image
    gray_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    gray_img, bin_img = PreProcessor.process(gray_img)
    gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()

    # Extract features of every line separately
    x = []
    for g, b in zip(gray_lines, bin_lines):
        f = FeatureExtractor([g], [b]).extract()
        x.append(f)

    # Return list of features for every line in the image
    return x


# =====================================================================
#
# Old unused functions
#
# =====================================================================


def process_test_case_old(path):
    features, labels, r = [], [], 0

    # Loop over every writer in the current test case
    # Should be 3 writers
    for root, dirs, files in os.walk(path):
        for d in dirs:
            print('    Processing writer', d, '...')
            x, y = process_writer_old(path + d + '/', d)
            features.extend(x)
            labels.extend(y)

    # Train the KNN model
    classifier = KNeighborsClassifier(1)
    classifier.fit(features, labels)

    # Loop over test images in the current test iteration
    # Should be 1 test image
    for root, dirs, files in os.walk(path):
        for filename in files:
            f = get_writing_features(path + filename)
            r = classifier.predict([f])[0]
            print('    Classifying test image \'%s\' as writer \'%s\'' % (path + filename, r))
            break
        break

    # Return classification
    return r


def process_writer_old(path, writer_id):
    x, y = [], []

    for root, dirs, files in os.walk(path):
        for filename in files:
            x.append(get_writing_features(path + filename))
            y.append(writer_id)

    return x, y


def get_writing_features(image_path):
    # Pre-processing
    gray_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    gray_img, bin_img = PreProcessor.process(gray_img)
    gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()

    # Feature extraction
    return FeatureExtractor(gray_lines, bin_lines).extract()


# =====================================================================
#
# Generate test iterations from IAM data set
#
# =====================================================================

if GENERATE_TEST_ITERATIONS:
    gen = TestGenerator()
    gen.generate(TEST_CASES_PATH, 10, 3, 2)

# =====================================================================
#
# Main
#
# =====================================================================

run()
print('-------------------------------')
print('Total elapsed time: %.2f seconds' % total_time)
print('Average test case time: %.2f seconds' % calculate_avg_test_case_time())
print('Classification accuracy: %d/%d' % calculate_accuracy())
print('-------------------------------')
print()

# print_wrong_test_cases()
# save_wrong_test_cases()
