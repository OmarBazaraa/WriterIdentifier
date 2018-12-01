import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.utils.utils import *
from src.segmentation.line_segmentor import LineSegmentor


class FeatureExtractor:

    def __init__(self, img, gray_img, bin_img):
        """
        Constructs a new feature extractor object for the given handwritten paragraph.

        :param img:         the original whole image.
        :param gray_img:    the pre-processed cropped gray image of the handwritten paragraph.
        :param bin_img:     the pre-processed cropped binary image of the handwritten paragraph.
        """

        # Store references to the page images.
        self.org_img = img
        self.gray_img = gray_img
        self.bin_img = bin_img

        # Initialize features list.
        self.features = []

        # Segment page into lines.
        segmentor = LineSegmentor(gray_img, bin_img)
        self.gray_lines, self.bin_lines = segmentor.segment()

    def extract(self):
        """
        Extracts the features of the handwritten text.

        :return:    the feature vector of the handwriting.
        """

        self.features.append(self.horizontal_run_length())
        self.features.append(self.average_line_height())
        self.features.extend(self.average_writing_width())

        return self.features

    #####################################################################

    def average_line_height(self):
        """
        Calculates and returns the average lines height relative to the
        original image height.

        :return:    the average relative lines height
        """

        avg_height = 0

        for line in self.gray_lines:
            h, w = line.shape
            avg_height += h

        avg_height /= len(self.gray_lines)

        return (avg_height / self.org_img.shape[0]) * 100

    #####################################################################

    def average_writing_width(self):
        """
        Calculates and returns the average writing width.

        :return:    real number representing the average writing width.
        """

        avg_width = 0
        avg_space = 0

        for line in self.bin_lines:
            w, s = FeatureExtractor.get_writing_width(line)
            avg_width += w
            avg_space += s

        avg_width /= len(self.bin_lines)
        avg_space /= len(self.bin_lines)

        return [avg_width, avg_space]

    @staticmethod
    def get_writing_width(bin_line):
        # Get line dimensions.
        height, width = bin_line.shape

        #
        # Get the row with maximum transitions.
        #
        row = trans_count = 0

        for i in range(height):
            # Count transitions in row i.
            cnt = 0
            for j in range(1, width):
                if bin_line[i][j] != bin_line[i][j - 1]:
                    cnt += 1

            # Update row index with maximum transitions.
            if cnt > trans_count:
                trans_count = cnt
                row = i

        #
        # Get white runs in the row with maximum transitions.
        #
        white_runs = []

        i = 0
        while i < width:
            # If black pixel then continue.
            if bin_line[row][i] != 0:
                i += 1
                continue

            # Count how many consecutive white pixels.
            j = i + 1
            while j < width and bin_line[row][j] == 0:
                j += 1

            # Append white run length.
            white_runs.append(j - i)

            i = j

        # Get the median white run.
        median_run = np.median(white_runs)

        # Get average space width between words.
        space_width = space_count = 0
        for r in white_runs:
            if r > 2 * median_run:
                space_width += r
                space_count += 1

        space_width /= space_count

        # Return writing width features.
        return median_run, space_width

    #####################################################################

    @staticmethod
    def get_lower_upper_baselines(gray_line, bin_line):
        height, width = gray_line.shape

        hist = np.sum(bin_line, axis=1, dtype=int) // 255
        total_black_pixels_count = np.sum(bin_line, dtype=int) // 255

        upper_baseline, lower_baseline = 0, 0

        iterations = 0

        error = 1e9

        ub = height // 2
        while ub >= 0:

            lb = height // 2
            while lb < height:
                e = 0

                for i in range(height):
                    iterations += 1

                    if ub <= i <= lb:
                        d = hist[i] - (total_black_pixels_count / (lb - ub + 1))
                        e += d * d
                    else:
                        d = hist[i]
                        e += d * d

                if e < error:
                    error = e
                    upper_baseline = ub
                    lower_baseline = lb

                lb += 1

            ub -= 1

        print('Iterations:', iterations)

        img = cv.cvtColor(gray_line, cv.COLOR_GRAY2BGR)
        cv.line(img, (0, upper_baseline), (width, upper_baseline), (0, 0, 255), 2)
        cv.line(img, (0, lower_baseline), (width, lower_baseline), (0, 0, 255), 2)
        display_image('Base lines', img, True)

        return upper_baseline, lower_baseline

    #####################################################################

    def horizontal_run_length(self):
        """
        Get the run length feature given gray lines.

        :return:    a number representing the horizontal run length.
        """
        ret = 0.0

        for line in self.gray_lines:
            dummy = 1

        return ret
