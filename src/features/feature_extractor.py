import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from ..segmentation.line_segmentor import LineSegmentor
from ..segmentation.word_segmentor import WordSegmentor


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
        Get the run length feature given gray lines.
        :return:    a number representing the horizontal run length.
        """
        ret = 0.0

        for line in self.gray_lines:
            dummy = 1

        return ret
