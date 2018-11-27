import os
import cv2 as cv
import numpy as np

from src.pre_processor import PreProcessor

images = []
labels = []

data_path = "../data/"

for root, dirs, files in os.walk(data_path + "/"):
    for filename in files:
        # Ignore gitignore.
        if filename[0] == '.':
            continue

        print(filename)

        # Read image in grayscale.
        gray_img = cv.imread(data_path + "/" + filename, cv.IMREAD_GRAYSCALE)

        # Pre process image.
        PreProcessor.pre_process(gray_img)

        # break

        # TODO get id of the writer.
