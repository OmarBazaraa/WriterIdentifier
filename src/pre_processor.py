import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola


class PreProcessor:
    @staticmethod
    def pre_process(gray_img):
        # Noise reduction
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        # FIXME: SAUVOLA
        # gray_img = threshold_sauvola(gray_img, window_size=5)

        _th, bin_img = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)

        # Crop image.
        return PreProcessor.__crop(gray_img, bin_img)

    @staticmethod
    def __crop(gray_img, bin_img):
        # Image dimensions
        h, w = gray_img.shape

        # Sobel horizontal edge detection.
        edge_img = cv.Sobel(bin_img, cv.CV_8U, 0, 1, ksize=5)

        # Horizontal histogram
        freq = np.sum(edge_img, axis=1) / 255

        # Threshold values
        THRESHOLD_HIGH = int(np.max(freq) * 0.76)
        THRESHOLD_LOW = 10
        UPPER_BOUND_OFFSET = 10

        # Upper and lower cropping bounds
        upper_bound = -1
        lower_bound = -1

        i = h - 1
        while i >= 0:
            if upper_bound < 0 and freq[i] > THRESHOLD_HIGH:  # Line end region.
                j = i
                while j > 0 and freq[j] > THRESHOLD_LOW:
                    j -= 1
                upper_bound = j - UPPER_BOUND_OFFSET
                i = j
            elif lower_bound < 0 and freq[i] > THRESHOLD_HIGH:
                j = i
                while j < h and freq[j] > THRESHOLD_LOW:
                    j += 1
                lower_bound = j
                break
            i -= 1

        # Plot and wait if wrong cropping occurs
        if upper_bound < 0 and lower_bound < 0:
            cv.imshow("R", edge_img)
            plt.figure()
            plt.plot([i for i in range(h)], freq)
            plt.show()

        # Skip upper white spaces
        while upper_bound > 0 and freq[upper_bound] == 0:
            upper_bound -= 1

        # Skip lower white spaces
        while lower_bound < h and freq[lower_bound] == 0:
            lower_bound += 1

        # Crop images
        gray_img = gray_img[lower_bound:upper_bound+1, :]
        bin_img = bin_img[lower_bound:upper_bound+1, :]

        # cv.imshow("R", edge_img)
        # plt.figure()
        # plt.plot([i for i in range(h)], freq)
        # plt.plot([lower_bound, upper_bound], [THRESHOLD_HIGH, THRESHOLD_HIGH])
        # plt.show()

        return gray_img, bin_img
