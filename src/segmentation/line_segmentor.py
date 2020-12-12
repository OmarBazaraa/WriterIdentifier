import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from utils.constants import *


class LineSegmentor:

    def __init__(self, gray_img: np.ndarray, bin_img: np.ndarray):
        """
        Constructs a new line segmentation object for the given handwritten paragraph image.

        :param gray_img:    the handwritten paragraph image in gray scale.
        :param bin_img:     the handwritten paragraph image after binarization.
        """

        # Store references to the page images.
        self.gray_img = gray_img
        self.bin_img = bin_img

        # Get horizontal histogram.
        self.hor_hist = np.sum(bin_img, axis=1, dtype=int) // 255

        # Get line density thresholds.
        self.threshold_high = int(np.max(self.hor_hist) // 3)
        self.threshold_low = 25

        # Initialize empty lists.
        self.peaks = []
        self.valleys = []
        self.lines_boundaries = []

        # Calculate peaks and valleys of the page.
        self.detect_peaks()
        self.avg_peaks_dist = int((self.peaks[-1] - self.peaks[0]) // len(self.peaks))
        self.detect_valleys()

        # Detect missing peaks and valleys in a second iteration.
        self.detect_missing_peaks_valleys()

        # Detect line boundaries.
        self.detect_line_boundaries()

    def segment(self):
        """
        Segments the handwritten paragraph into list of lines.

        :return:    two lists of lines:
                    one from the gray image and the other from the binary image.
        """

        # Illustrate line segmentation.
        self.display_segmentation()

        # Initialize lines lists.
        gray_lines, bin_lines = [], []

        # Loop on every line boundary.
        for l, u, r, d in self.lines_boundaries:
            # Crop gray line.
            g_line = self.gray_img[u:d + 1, l:r + 1]
            gray_lines.append(g_line)

            # Crop binary line.
            b_line = self.bin_img[u:d + 1, l:r + 1]
            bin_lines.append(b_line)

        # Return list of separated lines.
        return gray_lines, bin_lines

    def detect_peaks(self):
        """
        Detects the peak rows of the image and update self.peaks in correspondence.

        The peak rows are the ones with the highest black pixel density.
        """

        self.peaks = []

        i = 0
        while i < len(self.hor_hist):
            # If the black pixels density of the row is below than threshold
            # then continue to the next row.
            if self.hor_hist[i] < self.threshold_high:
                i += 1
                continue

            # Get the row with the maximum density from the following
            # probable row lines.
            peak_idx = i
            while i < len(self.hor_hist) and self.is_probable_peak(i):
                if self.hor_hist[i] > self.hor_hist[peak_idx]:
                    peak_idx = i
                i += 1

            # Add peak row index to the list.
            self.peaks.append(peak_idx)

    def detect_valleys(self):
        """
        Detects the valleys rows of the image and update self.valleys in correspondence.

        The valleys rows are the ones with the lowest black pixel density
        between two consecutive peaks.
        """

        self.valleys = [0]

        i = 1
        while i < len(self.peaks):
            u = self.peaks[i - 1]
            d = self.peaks[i]
            i += 1

            expected_valley = d - self.avg_peaks_dist // 2
            valley_idx = u

            while u < d:
                dist1 = np.abs(u - expected_valley)
                dist2 = np.abs(valley_idx - expected_valley)

                cond1 = self.hor_hist[u] < self.hor_hist[valley_idx]
                cond2 = self.hor_hist[u] == self.hor_hist[valley_idx] and dist1 < dist2

                if cond1 or cond2:
                    valley_idx = u

                u += 1

            self.valleys.append(valley_idx)

        self.valleys.append(len(self.hor_hist) - 1)

    def detect_missing_peaks_valleys(self):
        """
        Detects the missing peaks and valleys after the first detection trial
        using functions self.detect_peaks and self.detect_valleys.

        And updates self.peaks and self.valleys in correspondence.

        The missed peaks and valleys are probably because they are of shorter
        handwritten lines than the average lines length.
        """

        i = 1
        found = False

        while i < len(self.valleys):
            # Calculate distance between two consecutive valleys.
            up, down = self.valleys[i - 1], self.valleys[i]
            dis = down - up

            i += 1

            # If the distance is about twice the average distance between
            # two consecutive peaks, then it is most probable that we are missing
            # a line in between these two valleys.
            if dis < 1.5 * self.avg_peaks_dist:
                continue

            u = up + self.avg_peaks_dist
            d = min(down, u + self.avg_peaks_dist)

            while (d - u) * 2 > self.avg_peaks_dist:
                if self.is_probable_valley(u) and self.is_probable_valley(d):
                    peak = self.get_peak_in_range(u, d)
                    if self.hor_hist[peak] > self.threshold_low:
                        self.peaks.append(self.get_peak_in_range(u, d))
                        found = True

                u = u + self.avg_peaks_dist
                d = min(down, u + self.avg_peaks_dist)

        # Re-distribute peaks and valleys if new ones are found.
        if found:
            self.peaks.sort()
            self.detect_valleys()

    def detect_line_boundaries(self):
        """
        Detects handwritten lines of the image using the peaks and valleys.

        And updates self.lines_boundaries in correspondence.
        """

        # Get image dimensions.
        height, width = self.bin_img.shape

        self.lines_boundaries = []

        i = 1
        while i < len(self.valleys):
            u = self.valleys[i - 1]
            d = self.valleys[i]
            l = 0
            r = width - 1
            i += 1

            while u < d and self.hor_hist[u] == 0:
                u += 1
            while d > u and self.hor_hist[d] == 0:
                d -= 1

            ver_hist = np.sum(self.bin_img[u:d + 1, :], axis=0) // 255

            while l < r and ver_hist[l] == 0:
                l += 1
            while r > l and ver_hist[r] == 0:
                r -= 1

            self.lines_boundaries.append((l, u, r, d))

    def calc_average_line_slope(self) -> int:
        """
        Calculates the average range slope of the handwritten lines.

        See self.calc_range_slope for more information.

        :return:        the average range slope of the lines.
        """

        avg_slope = 0

        i = 1
        while i < len(self.valleys):
            u = self.valleys[i - 1]
            d = self.valleys[i]
            avg_slope += self.calc_range_slope(u, d)
            i += 1

        return int(avg_slope // (len(self.valleys) - 1))

    def calc_range_slope(self, up: int, down: int) -> int:
        """
        Calculates the range slope of black pixels density of the given range.

        Lets define the following quantities.

        let d(x)    be the black pixels density at row number x.
        let d'(x)   be the derivative of d(x) at row x.

        The range slope is calculated as:

        range slope = max(d'(i)) - min(d'(i)), where up <= i <= down

        :param up:      the upper row of the range.
        :param down:    the lower row of the range.
        :return:        the range slope.
        """
        max_der, min_der = -1e9, 1e9

        while up < down:
            up += 1
            val = self.hor_hist[up] - self.hor_hist[up - 1]
            max_der = max(max_der, val)
            min_der = min(min_der, val)

        return max_der - min_der

    def get_peak_in_range(self, up: int, down: int) -> int:
        """
        Finds the peak row in the given range from up to down inclusive.

        The peak row is the one with the highest black pixel density.

        :param up:      the upper row of the range.
        :param down:    the lower row of the range.
        :return:        the index of the peak row.
        """
        peak_idx = up

        while up < down:
            if self.hor_hist[up] > self.hor_hist[peak_idx]:
                peak_idx = up
            up += 1

        return peak_idx

    def is_probable_peak(self, row: int) -> bool:
        """
        Checks whether the given row is a probable peak row or not.

        The function depends on heuristics and is not deterministic.

        :param row:     the index of the row to check.
        :return:        boolean, whether the row is a probable peak or not.
        """
        width = 15

        for i in range(-width, width):
            if row + i < 0 or row + i >= len(self.hor_hist):
                continue
            if self.hor_hist[row + i] >= self.threshold_high:
                return True

        return False

    def is_probable_valley(self, row: int) -> bool:
        """
        Checks whether the given row is a probable valley row or not.

        The function depends on heuristics and is not deterministic.

        :param row:     the index of the row to check.
        :return:        boolean, whether the row is a probable valley or not.
        """
        width = 30
        count = 0

        for i in range(-width, width):
            if row + i < 0 or row + i >= len(self.hor_hist):
                return True
            if self.hor_hist[row + i] <= self.threshold_low:
                count += 1

        if count * 2 >= width:
            return True

        return False

    def display_segmentation(self):
        """
        Displays and visualizes segmentation steps.

        Used only while debugging.
        """

        # Display only in debugging mode.
        if not DEBUG_LINE_SEGMENTATION:
            return

        #
        # Draw bounding box around segmented lines.
        #
        img = cv.cvtColor(self.gray_img, cv.COLOR_GRAY2BGR)

        for l, u, r, d in self.lines_boundaries:
            cv.rectangle(img, (l, u), (r, d), (0, 0, 255), 2)

        display_image('Line Segmented Paragraph', img, False)

        #
        # Draw histogram.
        #
        plt.figure()
        plt.xlabel('Row index')
        plt.ylabel('Number of black pixels')
        plt.plot(list(range(len(self.hor_hist))), self.hor_hist)
        plt.plot([0, len(self.hor_hist)], [self.threshold_high, self.threshold_high], 'g--')

        # Draw peaks.
        for r in self.peaks:
            plt.plot(r, self.hor_hist[r], 'ro')
            plt.plot([r - self.avg_peaks_dist / 2, r + self.avg_peaks_dist / 2], [self.hor_hist[r], self.hor_hist[r]], 'r')

        # Draw valleys.
        for r in self.valleys:
            plt.plot(r, self.hor_hist[r], 'bs')

        # Draw probable missing valleys
        i = 1
        while i < len(self.valleys):
            dis = self.valleys[i] - self.valleys[i - 1]

            if dis > 1.8 * self.avg_peaks_dist:
                r = self.valleys[i]
                plt.plot(r - self.avg_peaks_dist, self.hor_hist[r], 'gs')

            i += 1

        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()
        cv.destroyAllWindows()
