#
# Paths
#
DATASET_PATH = './data/dataset/'
TESTCASES_PATH = './data/testcases/'
EXPECTED_RESULTS_FILENAME = 'results_expected.txt'
EXPECTED_RESULTS_PATH = TESTCASES_PATH + EXPECTED_RESULTS_FILENAME
PREDICTED_RESULTS_PATH = TESTCASES_PATH + 'results_predicted.txt'
WRONG_CASES_PATH = './data/wrong/'

# Make it true to generate test iterations from IAM data set.
GENERATE_TEST_ITERATIONS = False

# Make it true once to generate the pre processed images to save time.
GENERATE_PRE_PROCESSED_DATA = False

# Debugging flags.
DEBUG_PARAGRAPH_SEGMENTATION = False
DEBUG_LINE_SEGMENTATION = False
DEBUG_THINNING_ALGORITHM = False
DEBUG_SAVE_WRONG_TESTCASES = False
