import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from ..segmentation.word_segmentor import WordSegmentor
# from sklearn.neighbors import KernelDensity

from ..segmentation.line_segmentor import LineSegmentor


class FeatureExtractor:

    def __init__(self, img, gray_img, bin_img):
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
        # self.features.append(self.horizontal_run_length())
        self.features.append(self.average_line_height())
        return self.features

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

    def horizontal_run_length(self):
        """
        WIP
        Get the horizontal run length feature given binary lines.
        :return:    a number representing the horizontal run length.
        """
        freq = np.zeros((60,))

        # Calculate the frequency.
        for line in self.bin_lines:
            a = np.asarray(line)
            a[a == 0] = 22
            a[a == 1] = 0
            a[a == 22] = 1
            line_freq, bins = np.histogram(np.sum(a, axis=1), bins=60, density=True)
            freq += line_freq

        plt.plot(freq)
        plt.show()

        return freq

    def vertical_run_length(self):
        """
        WIP
        Get the vertical run length feature given binary lines.
        :return:    a number representing the horizontal run length.
        """
        freq = []

        # Calculate the frequency.
        for line in self.bin_lines:
            a = np.asarray(line)
            a[a == 0] = 22
            a[a == 1] = 0
            a[a == 22] = 1
            line_freq = np.sum(a, axis=0)[:]
            freq.extend(line_freq)
        print(len(freq))
        h, b = np.histogram(np.asarray(freq), bins=60)
        plt.plot(h)
        plt.show()

        return freq
