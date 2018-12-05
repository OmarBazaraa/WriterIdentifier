import xml.etree.ElementTree as ET
import os
import cv2 as cv
import numpy as np
from src.utils.utils import *
from ..pre_processor import PreProcessor


class IAMLoader:
    # Data set path
    processed_data_images_path = "../data/processed/form"
    processed_data_writer_ids = "../data/processed/writers_ids.txt"
    raw_data_path = "../data/raw/form"

    # Failed images.
    images_of_interest = ['h07-084.png', 'h07-080a.png', 'h07-075a.png', 'h07-078a.png', 'k03-164.png', 'k02-053.png',
                          'j07-005.png']

    # Top Writers only with freq of 10 pages and ignoring the wirter 0 who has 59 images.
    top_writers_ids = [
        # 150,
        # 151,
        # 152,
        # 153,
        # 154,
        # 384,
        # 551,
        # 552,
        588,
        635,
        670,
    ]

    @staticmethod
    def generate_processed_data():
        labels = {}
        # Walk on data set directory
        for root, dirs, files in os.walk(IAMLoader.raw_data_path + "/"):
            #
            # Loop on every file in the directory.
            #
            for filename in files:
                # Ignore gitignore file.
                if filename[0] == '.' or filename in IAMLoader.images_of_interest:
                    continue

                # Extract label (i.e. writer id).
                writer_id = IAMLoader.__get_writer_id(filename[:-4])
                labels[filename] = writer_id

                # Print image name.
                print(filename)

                # Read image in gray scale.
                img = cv.imread(IAMLoader.raw_data_path + "/" + filename, cv.IMREAD_GRAYSCALE)

                # Pre process image.
                gray_img, bin_img = PreProcessor.process(img, filename)

                # Save gray images to data processed folder.
                cv.imwrite(IAMLoader.processed_data_images_path + '/gray/' + filename, gray_img)
                cv.imwrite(IAMLoader.processed_data_images_path + '/bin/' + filename, bin_img)

            # Save the labels dict.
            file = open(IAMLoader.processed_data_writer_ids, "w")
            file.write(str(labels))
            file.close()

            # Break in order not to enter other dirs in the data/raw/form folder.
            break

    @staticmethod
    def __get_writer_id(filename):
        """
        Given a file names, find te file's writer id.
        :return: writer_id
        """
        tree = ET.parse("../data/raw/xml/" + filename + ".xml")
        root = tree.getroot()
        writer_id = int(root.attrib["writer-id"])
        return writer_id
