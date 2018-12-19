import cv2 as cv
import numpy as np


def display_image(name: str, img: np.ndarray, wait: bool = True) -> None:
    """
    Displays the given image in a new window
    and waits for a user keyboard input to close the window and return.

    To just display the image and return immediately, pass wait=False.

    :param name:    the name of the image to be displayed on the window.
    :param img:     the image to display.
    :param wait:    whether to wait for any key to close the image or not (defaults to true).
    :return:        nothing.
    """

    h, w = img.shape[0:2]
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w // 3, h // 3)
    cv.imshow(name, img)

    if wait:
        cv.waitKey(0)
        cv.destroyAllWindows()
