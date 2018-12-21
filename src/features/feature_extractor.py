import math
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage import feature

from src.utils.utils import *
from src.segmentation.line_segmentor import LineSegmentor


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

        self.features.extend(self.lbp_histogram())

        return self.features

    def lbp_histogram(self):
        hist = np.zeros(256)

        for i in range(len(self.bin_lines)):
            hist = FeatureExtractor.get_lbp_histogram(self.gray_lines[i], self.bin_lines[i], hist, True)

        hist /= np.mean(hist)

        # plt.figure()
        # plt.plot(list(range(len(hist))), hist)
        # plt.show()

        return hist

    @staticmethod
    def get_lbp_histogram(img, mask, hist=None, acc=True):
        # Get image dimensions
        height, width = img.shape

        # Initialize LBP image
        lbp = np.zeros((height, width), dtype=np.uint8)

        # Directions
        v = 3
        dx = [0, v, v, v, 0, -v, -v, -v]
        dy = [v, v, 0, -v, -v, -v, 0, v]

        # Loop over the 8 neighbors
        for i in range(8):
            view_shf = FeatureExtractor.shift(img, (dy[i], dx[i]))
            view_img = FeatureExtractor.shift(img, (-dy[i], -dx[i]))
            view_lbp = FeatureExtractor.shift(lbp, (-dy[i], -dx[i]))
            res = (view_img >= view_shf)
            view_lbp |= (res.view(np.uint8) << i)

        # Calculate LBP histogram of only black pixels
        hist = cv.calcHist([lbp], [0], mask, [256], [0, 256], hist, acc)
        hist = hist.ravel()

        return hist

    @staticmethod
    def shift(img, shift) -> np.ndarray:
        r, c = shift[0], shift[1]

        if r >= 0:
            ret = img[r:, :]
        else:
            ret = img[0:r, :]

        if c >= 0:
            ret = ret[:, c:]
        else:
            ret = ret[:, 0:c]

        return ret
