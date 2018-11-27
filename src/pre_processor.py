import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt


# from skimage.filters import threshold_sauvola


class PreProcessor:
    @staticmethod
    def pre_process(gray_img):
        # Reduce image noise.
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        # FIXME: SAUVOLA
        # gray_img = threshold_sauvola(gray_img, window_size=5)

        # Binarize the image.
        thresh, bin_img = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)

        # Crop image.
        return PreProcessor.__crop(gray_img, bin_img)

    @staticmethod
    def __crop(gray_img, bin_img):
        # Image dimensions.
        h, w = gray_img.shape

        # Sobel horizontal edge detection.
        edge_img = cv.Sobel(bin_img, cv.CV_8U, 0, 1, ksize=5)

        # Get horizontal histogram.
        freq = np.sum(edge_img, axis=1) / 255

        # Threshold values.
        threshold_high = int(np.max(freq) * 0.76)
        threshold_low = 10
        upper_bound_offset = 10

        # Lower and upper cropping boundaries.
        # The actual handwritten text is in [upper_line, lower_line]
        lower_line = -1
        upper_line = -1

        # Compute lower and upper boundaries.
        i = h - 1
        while i >= 0:
            if lower_line < 0 and freq[i] > threshold_high:     # Detect bottom black line
                j = i
                while j > 0 and freq[j] > threshold_low:
                    j -= 1
                lower_line = j - upper_bound_offset
                i = j
            elif upper_line < 0 and freq[i] > threshold_high:   # Detect upper black line
                j = i
                while j < h and freq[j] > threshold_low:
                    j += 1
                upper_line = j
                break
            i -= 1

        # Plot and terminate if wrong cropping occurs
        if lower_line < 0 and upper_line < 0:
            cv.imshow("Horizontal Edges", edge_img)
            plt.figure()
            plt.plot([i for i in range(h)], freq)
            plt.show()
            exit(0)

        # Skip bottom white spaces
        while lower_line > 0 and freq[lower_line] == 0:
            lower_line -= 1

        # Skip upper white spaces
        while upper_line < h and freq[upper_line] == 0:
            upper_line += 1

        # Crop images
        gray_img = gray_img[upper_line:lower_line + 1, :]
        bin_img = bin_img[upper_line:lower_line + 1, :]

        # cv.imshow("Horizontal Edges", edge_img)
        # plt.figure()
        # plt.plot([i for i in range(h)], freq)
        # plt.plot([lower_bound, upper_bound], [THRESHOLD_HIGH, THRESHOLD_HIGH])
        # plt.show()

        return gray_img, bin_img
