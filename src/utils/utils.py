import cv2 as cv
import numpy as np


def displayImage(name, img):
    h, w = img.shape
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w // 2, h // 2)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
