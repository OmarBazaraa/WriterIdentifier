import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.utils.utils import *


class LineSegmentor:
    @staticmethod
    def segment(gray_img, bin_img):
        # Get horizontal histogram.
        hor_hist = np.sum(bin_img, axis=1, dtype=int) // 255

        # Get line density threshold.
        threshold = int(np.max(hor_hist) // 3)

        # Detect peaks and valleys.
        peaks = LineSegmentor._detect_peaks(hor_hist, threshold)
        valleys = LineSegmentor._detect_valleys(hor_hist, peaks)

        # Calculate average distance between peaks.
        avg_dis = int((peaks[-1] - peaks[0]) // len(peaks))

        # Detect missing peaks and valleys in a second iteration.
        peaks, valleys = LineSegmentor._detect_missing_peaks_valleys(hor_hist, peaks, valleys, avg_dis)

        # Detect line boundaries.
        lines_boundaries = LineSegmentor._detect_line_boundaries(bin_img, hor_hist, valleys)

        #
        # Illustrate line segmentation.
        #
        if DEBUG_LINE_SEGMENTATION:
            # Draw bounding box around lines.
            img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

            for l, u, r, d in lines_boundaries:
                cv.rectangle(img, (l, u), (r, d), (0, 0, 255), 2)

            display_image('Line Segmented Paragraph', img, False)

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

            # Draw probable missing valleys
            # i = 1
            # while i < len(valleys):
            #     dis = valleys[i] - valleys[i - 1]
            #
            #     if dis > 1.8 * avg_dis:
            #         r = valleys[i]
            #         plt.plot(r - avg_dis, hor_hist[r], 'gs')
            #
            #     i += 1

            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()
            cv.destroyAllWindows()

        # Return list of separated lines.
        return LineSegmentor._crop_lines(gray_img, bin_img, lines_boundaries)

    @staticmethod
    def _detect_peaks(hor_hist, threshold):
        peaks = []

        i = 0
        while i < len(hor_hist):
            # Check if row is empty.
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

            # Add peak row index to the list.
            peaks.append(max_idx)

        return peaks

    @staticmethod
    def _detect_valleys(hor_hist, peaks):
        valleys = [0]

        i = 1
        while i < len(peaks):
            u = peaks[i - 1]
            d = peaks[i]
            i += 1
            min_idx = u
            while u < d:
                if hor_hist[u] < hor_hist[min_idx]:
                    min_idx = u
                u += 1
            valleys.append(min_idx)

        valleys.append(len(hor_hist) - 1)

        return valleys

    @staticmethod
    def _detect_missing_peaks_valleys(hor_hist, peaks, valleys, avg_line_dist):
        i = 1

        while i < len(valleys):
            dis = valleys[i] - valleys[i - 1]

            if dis > 1.8 * avg_line_dist:
                d = valleys[i]
                u = d - avg_line_dist
                p = LineSegmentor._detect_peak_in_range(hor_hist, u, d, 20)

                if p != -1:
                    peaks.append(p)

            i += 1

        peaks.sort()
        valleys = LineSegmentor._detect_valleys(hor_hist, peaks)

        return peaks, valleys

    @staticmethod
    def _detect_peak_in_range(hor_hist, up, down, threshold):
        range_len = down - up + 1
        derivative = np.zeros(range_len)

        max_der = -1e9
        min_der = 1e9
        peak_idx = up

        i = 1
        while i < range_len:
            r = up + i

            derivative[i] = hor_hist[r] - hor_hist[r - 1]

            max_der = max(max_der, derivative[i])
            min_der = min(min_der, derivative[i])

            if hor_hist[r] > hor_hist[peak_idx]:
                peak_idx = r

            i += 1

        # print('Max diff', max_der - min_der)

        # plt.figure()
        # plt.plot(list(range(len(derivative))), derivative)
        # plt.draw()
        # plt.waitforbuttonpress(0)
        # plt.close()

        if max_der - min_der < threshold:
            return -1

        return peak_idx

    @staticmethod
    def _detect_line_boundaries(bin_img, hor_hist, valleys):
        # Get image dimensions.
        height, width = bin_img.shape

        lines_boundaries = []

        i = 1
        while i < len(valleys):
            u = valleys[i - 1]
            d = valleys[i]
            l = 0
            r = width - 1
            i += 1

            while u < d and hor_hist[u] == 0:
                u += 1
            while d > u and hor_hist[d] == 0:
                d -= 1

            ver_hist = np.sum(bin_img[u:d + 1, :], axis=0) // 255

            while l < r and ver_hist[l] == 0:
                l += 1
            while r > l and ver_hist[r] == 0:
                r -= 1

            lines_boundaries.append((l, u, r, d))

        return lines_boundaries

    @staticmethod
    def _calc_average_line_height(lines_boundaries):
        height = 0
        for l, u, r, d in lines_boundaries:
            height += d - u
        return height / len(lines_boundaries)

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
