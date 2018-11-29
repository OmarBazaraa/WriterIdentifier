import cv2 as cv
import numpy as np
import mxnet as mx

DEBUG_PARAGRAPH_SEGMENTATION = False
DEBUG_LINE_SEGMENTATION = False


def display_image(name, img, wait=True):
    h, w = img.shape[0:2]
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w // 2, h // 2)
    cv.imshow(name, img)

    if wait:
        cv.waitKey(0)
        cv.destroyAllWindows()


'''
Returns the index of a compatible gpu if found, None otherwise
'''
def gpu_device(gpu_number=0):
    try:
        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
    except mx.MXNetError:
        return None
    return mx.gpu(gpu_number)
