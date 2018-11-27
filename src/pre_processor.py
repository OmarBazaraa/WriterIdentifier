import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola


class PreProcessor:
    @staticmethod
    def pre_process(gray_img):
        UPPER_BOUND_OFFSET = 10

        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        # FIXME: SAUVOLA
        # gray_img = threshold_sauvola(gray_img, window_size=5)

        _th, bin_img = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)

        # Crop image.
        l, u = PreProcessor.__crop(bin_img)
        gray_img = gray_img[l:u - UPPER_BOUND_OFFSET, :]
        bin_img = bin_img[l:u - UPPER_BOUND_OFFSET, :]

    @staticmethod
    def __crop(bin_img):

        # Sobel horizontal edge detection.
        edge_img = cv.Sobel(bin_img, cv.CV_8U, 0, 1, ksize=5)

        h, w = edge_img.shape

        freq = np.sum(edge_img, axis=1) / 255

        THRESHOLD_HIGH = int(np.max(freq) * 0.76)
        THRESHOLD_LOW = 10

        upper_bound = -1
        lower_bound = -1
        i = h - 1
        while i >= 0:
            if upper_bound < 0 and freq[i] > THRESHOLD_HIGH:  # Line end region.
                j = i
                while j > 0 and freq[j] > THRESHOLD_LOW:
                    j -= 1
                upper_bound = j
                i = j
            elif lower_bound < 0 and freq[i] > THRESHOLD_HIGH:
                j = i
                while j < h and freq[j] > THRESHOLD_LOW:
                    j += 1
                lower_bound = j
                break
            i -= 1

        if upper_bound < 0 and lower_bound < 0:
            # cv.imshow("R", edge_img)
            plt.figure()
            plt.plot([i for i in range(h)], freq)
            plt.show()

        # cv.imshow("R", edge_img)
        # plt.figure()
        # plt.plot([i for i in range(h)], freq)
        # plt.plot([lower_bound, upper_bound], [THRESHOLD_HIGH, THRESHOLD_HIGH])
        # plt.show()

        return lower_bound, upper_bound
