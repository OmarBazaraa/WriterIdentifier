import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from skimage.filters import threshold_sauvola

from src.utils.utils import *


class PreProcessor:
    @staticmethod
    def process(gray_img, filename):
        # Reduce image noise.
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        # FIXME: SAUVOLA
        # gray_img = threshold_sauvola(gray_img, window_size=5)

        # Initial cropping.
        l_padding = 150
        r_padding = 50
        gray_img = gray_img[:, l_padding:-r_padding]

        # Binarize the image.
        thresh, bin_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Crop page header and footer and keep only the handwritten area.
        gray_img, bin_img = PreProcessor._crop_paragraph(gray_img, bin_img, filename)

        # Return pre processed images.
        return gray_img, bin_img

    @staticmethod
    def _crop_paragraph(gray_img, bin_img, filename):
        # Get image dimensions.
        height, width = gray_img.shape

        # Find all contours in the page.
        _, contours, hierarchy = cv.findContours(bin_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        # Minimum contour width to be considered as the black separator line
        threshold_width = 1000

        # Page paragraph boundaries.
        up, down, left, right = 0, height - 1, 0, width - 1

        # Detect the main horizontal black separator lines of the IAM handwriting forms.
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)

            if w < threshold_width:
                continue

            if y < height // 2:
                up = max(up, y + h)
            else:
                down = min(down, y)

        # Apply median filter and erosion to remove noise and dots.
        kernel = np.ones((3, 3), np.uint8)
        eroded_img = cv.erode(bin_img, kernel, iterations=1)

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

        # Give some padding to the paragraph
        padding = 10
        left -= padding
        up -= padding
        right += padding
        down += padding

        # Crop images.
        gray_img = gray_img[up:down + 1, left:right + 1]
        bin_img = bin_img[up:down + 1, left:right + 1]

        if DEBUG_PARAGRAPH_SEGMENTATION:
            # Display bounding box on the handwritten paragraph
            img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
            cv.rectangle(img, (left, up), (right, down), (0, 0, 255), 3)
            display_image('Image', img)

            cv.imwrite("../data/output/" + filename, gray_img)

        # Return the handwritten paragraph
        return gray_img, bin_img

    @staticmethod
    def _crop(gray_img, bin_img):
        # Get image dimensions.
        h, w = gray_img.shape

        # Sobel horizontal edge detection.
        edge_img = cv.Sobel(bin_img, cv.CV_8U, 0, 1, ksize=5)

        # Apply median filter to remove noise.
        edge_img = cv.medianBlur(edge_img, 7)

        # Get horizontal and vertical histograms.
        hor_hist = np.sum(edge_img, axis=1) / 255
        ver_hist = np.sum(edge_img, axis=0) / 255

        # Threshold values.
        threshold_high = int(np.max(hor_hist) * 0.76)
        threshold_low = 10
        lower_line_offset = 20
        upper_line_offset = 10

        # Page boundaries.
        up, down, left, right = -1, -1, 0, w - 1

        # Detect top and bottom black lines between which the handwritten area is located.
        i = h - 1
        while i >= 0:
            # Continue if not a black row
            if hor_hist[i] < threshold_high:
                i -= 1
                continue

            if down < 0:
                # Detect bottom black line
                j = i
                while j > 0 and hor_hist[j] > threshold_low:
                    j -= 1
                down = j - lower_line_offset
                i = j
            elif up < 0:
                # Detect top black line
                j = i
                while j < h and hor_hist[j] > threshold_low:
                    j += 1
                up = j + upper_line_offset
                break

        # Plot and terminate if wrong cropping occurs.
        if up < 0 and down < 0:
            display_image("Horizontal Edges", edge_img, False)
            plt.figure()
            plt.plot(list(range(h)), hor_hist)
            plt.show()
            exit(0)

        # Detect page white padding.
        while left < right and ver_hist[left] == 0:
            left += 1
        while right > left and ver_hist[right] == 0:
            right -= 1
        while up < down and hor_hist[up] == 0:
            up += 1
        while down > up and hor_hist[down] == 0:
            down -= 1

        # Crop images.
        gray_img = gray_img[up:down + 1, left:right + 1]
        bin_img = bin_img[up:down + 1, left:right + 1]

        #
        display_image('Img', bin_img)

        # Plot the histogram.
        # cv.imshow("Horizontal Edges", edge_img)
        # plt.figure()
        # plt.plot([i for i in range(h)], freq)
        # plt.plot([upper_line, lower_line], [threshold_high, threshold_high])
        # plt.show()

        return gray_img, bin_img
