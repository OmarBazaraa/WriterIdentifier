import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from utils.constants import *


class PreProcessor:
    @staticmethod
    def process(gray_img: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Pre-processes the IAM form image and extracts the handwritten paragraph only.

        :param gray_img:    the IAM form image to be processed.
        :return:            pre-processed gray and binary images of the handwritten paragraph.
        """

        # Resize image.
        # height, width = gray_img.shape
        # limit = 1000
        # if height > limit:
        #     ratio = limit / height
        #     gray_img = cv.resize(gray_img, (0, 0), fx=ratio, fy=ratio)

        # Reduce image noise.
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        # Initial cropping.
        l_padding = 150
        r_padding = 50
        gray_img = gray_img[:, l_padding:-r_padding]

        # Binarize the image.
        thresh, bin_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Crop page header and footer and keep only the handwritten area.
        gray_img, bin_img = PreProcessor._crop_paragraph(gray_img, bin_img)

        # Return pre processed images.
        return gray_img, bin_img

    @staticmethod
    def _crop_paragraph(gray_img: np.ndarray, bin_img: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Detects the bounding box of the handwritten paragraph of the given IAM form image
        and returns a cropped image of it.

        :param gray_img:    the IAM form image to be processed.
        :param bin_img:     binarized IAM form image to be processed.
        :return:            cropped gray and binary images of the handwritten paragraph.
        """

        # Get image dimensions.
        height, width = gray_img.shape

        # Find all contours in the page.
        contours, hierarchy = cv.findContours(bin_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        # Minimum contour width to be considered as the black separator line.
        threshold_width = 1000
        line_offset = 10

        # Page paragraph boundaries.
        up, down, left, right = 0, height - 1, 0, width - 1

        # Detect the main horizontal black separator lines of the IAM handwriting forms.
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)

            if w < threshold_width:
                continue

            if y < height // 2:
                up = max(up, y + h + line_offset)
            else:
                down = min(down, y - line_offset)

        # Apply erosion to remove noise and dots.
        kernel = np.ones((3, 3), np.uint8)
        eroded_img = cv.erode(bin_img, kernel, iterations=2)

        # Get horizontal and vertical histograms.
        hor_hist = np.sum(eroded_img, axis=1) / 255
        ver_hist = np.sum(eroded_img, axis=0) / 255

        # Detect paragraph white padding.
        while left < right and ver_hist[left] == 0:
            left += 1
        while right > left and ver_hist[right] == 0:
            right -= 1
        while up < down and hor_hist[up] == 0:
            up += 1
        while down > up and hor_hist[down] == 0:
            down -= 1

        # Display bounding box on the handwritten paragraph.
        if DEBUG_PARAGRAPH_SEGMENTATION:
            img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
            cv.rectangle(img, (left, up), (right, down), (0, 0, 255), 3)
            display_image('Handwritten Paragraph', img)

        # Crop images.
        gray_img = gray_img[up:down + 1, left:right + 1]
        bin_img = bin_img[up:down + 1, left:right + 1]

        # Return the handwritten paragraph
        return gray_img, bin_img
