import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.utils.utils import *


class LineSegmentor:
    @staticmethod
    def segment(gray_img, bin_img):
        # Get image dimensions.
        height, width = gray_img.shape

        # Get horizontal histogram.
        hor_hist = np.sum(bin_img, axis=1) // 255

        # Get line density threshold.
        threshold = int(np.max(hor_hist) // 3)

        # Detect peak rows.
        peaks = []
        i = 0
        while i < len(hor_hist):
            # Check if line is not empty
            if hor_hist[i] < threshold:
                i += 1
                continue

            # Get peak row.
            j = max_idx = i
            while j < len(hor_hist) and LineSegmentor._is_probable_line(hor_hist, j, threshold):
                if hor_hist[j] > hor_hist[max_idx]:
                    max_idx = j
                j += 1
            i = j

            # Add peak row index to the list
            peaks.append(max_idx)

        # Calculate average distance between lines.
        avg_dis = 0
        i = 1
        while i < len(peaks):
            avg_dis += peaks[i] - peaks[i - 1]
            i += 1
        avg_dis /= len(peaks)

        # Detect valleys between consecutive peaks.
        valleys = [0]
        i = 1
        while i < len(peaks):
            u = peaks[i - 1]
            d = peaks[i]
            i += 1
            j = min_idx = u
            while j < d:
                if hor_hist[j] < hor_hist[min_idx]:
                    min_idx = j
                j += 1
            valleys.append(min_idx)
        valleys.append(len(hor_hist) - 1)

        # Detect line boundaries.
        lines_boundaries = []
        i = 1
        while i < len(valleys):
            u = valleys[i - 1]
            d = valleys[i]
            l = 0
            r = width - 1

            while u < d and hor_hist[u] == 0:
                u += 1
            while d > u and hor_hist[d] == 0:
                d -= 1

            ver_hist = np.sum(bin_img[u:d + 1, :], axis=0) / 255

            while l < r and ver_hist[l] == 0:
                l += 1
            while r > l and ver_hist[r] == 0:
                r -= 1

            lines_boundaries.append((l, u, r, d))

            i += 1

        #
        # Illustrate line segmentation.
        #
        if DEBUG_LINE_SEGMENTATION:
            # Draw bounding box around lines.
            img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

            for l, u, r, d in lines_boundaries:
                cv.rectangle(img, (l, u), (r, d), (0, 0, 255), 2)

            display_image('Binary Paragraph', img, False)

            # Draw histogram
            plt.figure()
            plt.xlabel('Row index')
            plt.ylabel('Number of black pixels')
            plt.plot(list(range(len(hor_hist))), hor_hist)
            plt.plot([0, len(hor_hist)], [threshold, threshold], 'g--')

            # Draw peaks.
            for r in peaks:
                plt.plot(r, hor_hist[r], 'ro')
                plt.plot([r - avg_dis / 2, r + avg_dis / 2], [hor_hist[r], hor_hist[r]], 'r')

            # Draw valleys.
            for r in valleys:
                plt.plot(r, hor_hist[r], 'bs')

            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()
            cv.destroyAllWindows()

        # Return list of separated lines.
        return LineSegmentor._crop_lines(gray_img, bin_img, lines_boundaries)

    @staticmethod
    def _is_probable_line(hor_hist, row, threshold):
        width = 15

        for i in range(-width, width):
            if row + i < 0 or row + i >= len(hor_hist):
                continue
            if hor_hist[row + i] >= threshold:
                return True

        return False

    @staticmethod
    def _crop_lines(gray_img, bin_img, lines_boundaries):
        return None, None
