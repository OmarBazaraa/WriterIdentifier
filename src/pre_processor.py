import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.segmentation.paragraph_segmentation_dcnn import make_cnn as ParagraphSegmentationNet
from src.segmentation.paragraph_segmentation_dcnn import paragraph_segmentation_transform
from src.utils.iam_dataset import IAMDataset, crop_handwriting_page
from src.utils.utils import *


# from skimage.filters import threshold_sauvola


class PreProcessor:
    @staticmethod
    def process(gray_img):
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
        gray_img, bin_img = PreProcessor.__crop(gray_img, bin_img)

        # Return pre processed images.
        return gray_img, bin_img

    @staticmethod
    def __crop(gray_img, bin_img):
        # Get image dimensions.
        h, w = gray_img.shape

        # Sobel horizontal edge detection.
        edge_img = cv.Sobel(bin_img, cv.CV_8U, 0, 1, ksize=5)

        # Apply median filter to remove noise.
        edge_img = cv.medianBlur(edge_img, 7)

        # Get horizontal and vertical histograms.
        hor_hist = np.sum(edge_img, axis=1) / 255
        ver_hist = np.sum(edge_img, axis=0) / 255

        '''
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

        # Plot the histogram.
        # cv.imshow("Horizontal Edges", edge_img)
        # plt.figure()
        # plt.plot([i for i in range(h)], freq)
        # plt.plot([upper_line, lower_line], [threshold_high, threshold_high])
        # plt.show()
        '''

        # Initialization for the model.
        # Check if a gpu is available.
        form_size = (1120, 800)
        segmented_paragraph_size = (800, 800)

        if gpu_device():
            ctx = mx.gpu(0)
        else:
            ctx = mx.cpu()

        # Create the paragraph segmentation using DCNN model.
        paragraph_segmentation_net = ParagraphSegmentationNet(ctx)
        paragraph_segmentation_net.load_parameters("../models/paragraph_segmentor/paragraph_segmentation2.params", ctx)

        # Resize the image before feeding it to the model.
        resized_image = paragraph_segmentation_transform(gray_img, image_size=form_size)
        # Page bounding box.
        paragraph_bb = paragraph_segmentation_net(resized_image.as_in_context(ctx))
        # Crop the handwritten paragraph.
        paragraph_segmented_image = crop_handwriting_page(gray_img, paragraph_bb[0].asnumpy(),
                                                          image_size=segmented_paragraph_size)

        display_image("Paragaph", paragraph_segmented_image, True)

        return gray_img, bin_img
