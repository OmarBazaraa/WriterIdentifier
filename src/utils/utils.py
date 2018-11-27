import cv2 as cv
import numpy as np


def display_image(name, img, wait=True):
    h, w = img.shape
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w // 2, h // 2)
    cv.imshow(name, img)

    if wait:
        cv.waitKey(0)
        cv.destroyAllWindows()
