import os
import shutil
import cv2 as cv
import numpy as np

from utils.constants import *


def display_image(name: str, img: np.ndarray, wait: bool = True) -> None:
    """
    Displays the given image in a new window
    and waits for a user keyboard input to close the window and return.

    To just display the image and return immediately, pass wait=False.

    :param name:    the name of the image to be displayed on the window.
    :param img:     the image to display.
    :param wait:    whether to wait for any key to close the image or not (defaults to true).
    :return:        nothing.
    """

    h, w = img.shape[0:2]
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w // 3, h // 3)
    cv.imshow(name, img)

    if wait:
        cv.waitKey(0)
        cv.destroyAllWindows()


def copy_file(src: str, dst: str) -> None:
    """
    Copies a file from its source path to the new given destination.
    
    :param src: the source path of the file.
    :param dst: the destination path of the file.
    """

    directory = os.path.dirname(dst)

    if not os.path.exists(directory):
        os.makedirs(directory)

    shutil.copyfile(src, dst)


def chunk(l: list, n: int) -> list:
    ret = []
    for i in range(0, len(l), n):
        ret.append(l[i:i + n])
    return ret


def calculate_accuracy():
    # Check if the expected results exists
    if not os.path.exists(EXPECTED_RESULTS_PATH):
        print('>> Please add \'%s\' with the expected results of all the test images' % EXPECTED_RESULTS_PATH)
        return -1, 0

    # Read results
    with open(PREDICTED_RESULTS_PATH) as f:
        predicted_res = f.read().splitlines()
    with open(EXPECTED_RESULTS_PATH) as f:
        expected_res = f.read().splitlines()

    # Calculate accuracy
    cnt = 0
    for i in range(len(predicted_res)):
        if predicted_res[i] == expected_res[i]:
            cnt += 1

    # Return accuracy
    return cnt, len(predicted_res)


def list_test_directory_content(directory, output_file):
    file = open(output_file, 'w+')

    def rec(path, ind=0):

        for root, dirs, files in os.walk(path):
            for d in dirs:
                file.write((' ' * ind) + d + '/\n')
                print((' ' * ind) + d + '/')
                rec(path + d + '/', ind + 4)
            for f in files:
                file.write(' ' * ind + f + '\n')
                print(' ' * ind + f)
            break

    rec(directory, 0)
    file.close()


def print_wrong_testcases():
    # Check if the expected results exists
    if not os.path.exists(EXPECTED_RESULTS_PATH):
        print('Please add \'%s\' with the expected results of all the test images' % EXPECTED_RESULTS_PATH)
        return

    # Read results
    with open(PREDICTED_RESULTS_PATH) as f:
        predicted_res = f.read().splitlines()
    with open(EXPECTED_RESULTS_PATH) as f:
        expected_res = f.read().splitlines()

    # Print wrong classifications
    for i in range(len(predicted_res)):
        if predicted_res[i] == expected_res[i]:
            continue

        print('Wrong classification in iteration: %s (output: %s, expected: %s)' %
              (format(i, '03d'), predicted_res[i], expected_res[i]))


def save_wrong_testcases():
    # Check if the expected results exists
    if not os.path.exists(EXPECTED_RESULTS_PATH):
        print('Please add \'%s\' with the expected results of all the test images' % EXPECTED_RESULTS_PATH)
        return

    # Read results
    with open(PREDICTED_RESULTS_PATH) as f:
        predicted_res = f.read().splitlines()
    with open(EXPECTED_RESULTS_PATH) as f:
        expected_res = f.read().splitlines()

    # Create wrong classified data directory
    if not os.path.exists(WRONG_CASES_PATH):
        os.makedirs(WRONG_CASES_PATH)

    # Open expected output text file
    file = open(WRONG_CASES_PATH + 'output.txt', 'a+')

    # Get number of previously wrong classified iterations
    k = 0
    for root, dirs, files in os.walk(WRONG_CASES_PATH):
        k = len(dirs)
        break

    # Save wrong classifications
    for i in range(len(predicted_res)):
        if predicted_res[i] == expected_res[i]:
            continue

        # Copy
        src_path = TESTCASES_PATH + format(i, '03d') + '/'
        dst_path = WRONG_CASES_PATH + format(k, '03d') + '/'
        shutil.copytree(src_path, dst_path)
        k += 1

        # Write expected result
        file.write(expected_res[i] + '\n')

    # Close file
    file.close()
