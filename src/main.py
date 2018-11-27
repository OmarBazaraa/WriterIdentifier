import os
import cv2 as cv
import numpy as np

from src.pre_processor import PreProcessor
from src.segmentation.line_segmentor import LineSegmentor
from src.feature_extractor import FeatureExtractor

featuers = []
labels = []

data_path = "../data/"

for root, dirs, files in os.walk(data_path + "/"):
    for filename in files:
        # Ignore gitignore file
        if filename[0] == '.':
            continue

        print(filename)

        # Read image in grayscale.
        gray_img = cv.imread(data_path + "/" + filename, cv.IMREAD_GRAYSCALE)

        # Pre process image.
        gray_img, bin_img = PreProcessor.pre_process(gray_img)

        # Line segment
        (gray_lines, bin_lines) = LineSegmentor.segment(gray_img, bin_img)

        # Extract features.
        featuers.append(FeatureExtractor.extract_features(gray_img, bin_img))

        # TODO Extract label.



        # break

        # TODO get id of the writer.
