import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.utils.utils import *


class LineSegmentor:

    @staticmethod
    def segment(gray_img, bin_img):
        # Get image dimensions.
        h, w = gray_img.shape

        # Get horizontal histogram.
        freq = np.sum(bin_img, axis=1) / 255

        # Threshold values.
        threshold_val = 10  # Maximum number of black pixels in a row to consider it blank row
        threshold_cnt = 10  # Minimum number of contiguous blank rows to consider it blank line

        # List of blank rows
        blank_rows = [0]

        # Detect blank contiguous rows.
        i = 0
        while i < h:
            # Check if line is not empty
            if freq[i] > threshold_val:
                i += 1
                continue

            # Get contiguous blank (white) rows.
            j = min_idx = i
            while j < h and freq[j] <= threshold_val:
                if freq[j] < freq[min_idx]:
                    min_idx = j
                j += 1

            # Add blank line if its height is not less than the threshold
            if j - i >= threshold_cnt:
                blank_rows.append(min_idx)

            i = j

        blank_rows.append(h - 1)

        print(blank_rows)

        for r in blank_rows:
            cv.line(bin_img, (0, r), (w, r), (255, 255, 0), 5)

        display_image('Img', bin_img, False)

        plt.figure()
        plt.plot(list(range(h)), freq)
        plt.show()

        return None, None
