import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from ..segmentation.word_segmentor import WordSegmentor
# from sklearn.neighbors import KernelDensity
from src.features.thinning import zhangSuen
from src.utils.utils import *
from ..segmentation.line_segmentor import LineSegmentor


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

        # self.features.append(self.horizontal_run_length()) # WIP
        # self.features.append(self.vertical_run_length()) # WIP
        # self.features.append(self.average_line_height())
        # self.features.extend(self.average_writing_width())
        # self.features.extend(self.average_contours_properties())
        self.features.extend(self.get_gmm_writer_features(14))

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

        if space_count > 0:  # FIXME @OmarBazaraa
            space_width /= space_count

        # Return writing width features.
        return median_run, space_width

    #####################################################################

    def average_contours_properties(self):
        prop = FeatureExtractor.get_contours_properties(self.bin_lines[0])

        for i in range(1, len(self.bin_lines)):
            f = FeatureExtractor.get_contours_properties(self.bin_lines[i])
            prop = np.add(prop, f)

        return np.multiply(prop, len(self.bin_lines))

    @staticmethod
    def get_contours_properties(bin_line):
        # Find all contours in the line.
        _, contours, hierarchy = cv.findContours(bin_line, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        aspect_ratio = 0
        extent = 0
        solidity = 0
        equi_diameter = 0

        for cnt in contours:
            if len(cnt) < 3:
                continue

            x, y, w, h = cv.boundingRect(cnt)
            rect_area = w * h
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            area = cv.contourArea(cnt)

            aspect_ratio += float(w) / h
            extent += float(area) / rect_area
            solidity += float(area) / hull_area
            equi_diameter += np.sqrt(4 * area / np.pi)

        # # Average distance between 2 contours.
        # # From paper found in "https://link.springer.com/chapter/10.1007/3-540-44887-X_79"
        # # Sort contours according to their positions from left to right.
        # bounding_boxes = [cv.boundingRect(c) for c in contours]
        # (cnts, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
        #                                      key=lambda b: b[1][0], reverse=False))
        # # Calculate the average distance between 2 contours.
        # dists = [bounding_boxes[i + 1][0] - (bounding_boxes[i][0])for i in range(len(cnts) - 1)]
        # if len(dists) == 0:
        #     dists = [0]

        if len(contours) > 0:
            aspect_ratio /= len(contours)
            extent /= len(contours)
            solidity /= len(contours)
            equi_diameter /= len(contours)

        return [aspect_ratio, extent, solidity, equi_diameter]

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

        # print('Iterations:', iterations)

        img = cv.cvtColor(gray_line, cv.COLOR_GRAY2BGR)
        cv.line(img, (0, upper_baseline), (width, upper_baseline), (0, 0, 255), 2)
        cv.line(img, (0, lower_baseline), (width, lower_baseline), (0, 0, 255), 2)
        display_image('Base lines', img, True)

        return upper_baseline, lower_baseline

    #####################################################################

    def horizontal_run_length(self):
        """
        WIP
        Get the horizontal run length feature given binary lines.
        :return:    a pdf representing the horizontal run length.
        """
        freq = np.zeros((1000,))

        # Calculate the frequency.
        for line in self.bin_lines:
            a = np.asarray(line).copy()
            # Swap white with black.
            a[a == 0] = 3
            a[a == 1] = 0
            a[a == 3] = 1
            line_freq, bins = np.histogram(np.sum(a, axis=1), bins=1000, density=True)
            freq += line_freq

        return np.argmax(freq)

    def vertical_run_length(self):
        """
        WIP
        Get the vertical run length feature given binary lines.
        :return:    a pdf representing the horizontal run length.
        """
        freq = np.zeros((1000,))

        # Calculate the frequency.
        for line in self.bin_lines:
            a = np.asarray(line).copy()
            # Swap white with black.
            a[a == 0] = 3
            a[a == 1] = 0
            a[a == 3] = 1
            line_freq, bins = np.histogram(np.sum(a, axis=0), bins=1000, density=True)
            freq += line_freq

        return np.argmax(freq)

    #####################################################################

    def get_slant(self):
        return None

    #####################################################################

    # For GMM Model features.
    def get_gmm_writer_features(self, sliding_window_width):
        # Loop over each line.
        lines_feauters = []
        for line in self.bin_lines:
            # Apply thinning algorithm.
            skeleton_image = zhangSuen(line)

            # Features
            windows_features = []

            # For every column of pixels apply the sliding window.
            t = time.clock()
            for i in range(np.asarray(line).shape[1] - sliding_window_width):
                window_features = []
                # Get the window of image.
                window = line[:, i:i + sliding_window_width]

                # Get number of black pixels. F1
                num_black_pixels = cv.countNonZero(window)

                # Find all contours in the line.
                _, contours, hierarchy = cv.findContours(window, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                # Ignore window without any contours.
                if len(contours) == 0:
                    continue

                # Get the moments.
                mu = [cv.moments(cnt, False) for cnt in contours]

                # Calculate centre of gravity. F2
                centre_of_gravity = np.mean(
                    [(mu[i]['m10'] / (mu[i]['m00'] + 0.001), mu[i]['m01'] / (mu[i]['m00'] + 0.001)) for i in
                     range(len(mu))],
                    axis=0)

                # Get the position of the upper and the lower contour. F3
                # x, y, w, h = cv.boundingRect(cnt).
                rects = [cv.boundingRect(cnt) for cnt in contours]
                up, lw = (np.min([rect[0] for rect in rects]), np.max([rect[0] for rect in rects]))

                # Calculate second order moments. F4
                second_order_moments = np.mean(
                    [(mu[i]['m02'], mu[i]['m20']) for i in
                     range(len(mu))],
                    axis=0)

                # TODO add left features. F5, F6, F7, F8, F9.
                window_features.extend([num_black_pixels, up, lw])
                window_features.extend(centre_of_gravity)
                window_features.extend(second_order_moments)

                # Append to windows_features.
                windows_features.append(window_features)

            lines_feauters.extend(np.mean(windows_features, axis=0))
        return lines_feauters
