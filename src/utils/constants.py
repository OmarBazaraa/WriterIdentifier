#
# Paths
#
DATASET_PATH = '../data/dataset/'
TEST_CASES_PATH = '../data/testcases/'
RESULTS_PATH = '../data/results.txt'
ELAPSED_TIME_PATH = '../data/time.txt'
EXPECTED_RESULTS_FILENAME = 'expected.txt'
EXPECTED_RESULTS_PATH = TEST_CASES_PATH + EXPECTED_RESULTS_FILENAME
WRONG_CASES_PATH = '../data/wrong/'

# Make it true to generate test iterations from IAM data set.
GENERATE_TEST_ITERATIONS = False

# Make it true once to generate the pre processed images to save time.
GENERATE_PRE_PROCESSED_DATA = False

# Debugging flags.
DEBUG_PARAGRAPH_SEGMENTATION = False
DEBUG_LINE_SEGMENTATION = False
DEBUG_THINNING_ALGORITHM = False
