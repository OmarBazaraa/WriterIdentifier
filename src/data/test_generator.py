import random

from utils.utils import *


class TestGenerator:
    dataset_path = DATASET_PATH + 'forms/'
    dataset_meta_file = DATASET_PATH + 'meta.txt'

    writers = {}
    writers_ids = []
    output_file = None

    def __init__(self):
        # Read IAM meta data file
        meta = open(self.dataset_meta_file, 'r')

        # Read all images with their writers
        for line in meta:
            # Skip comment lines
            if line[0] == '#':
                continue

            # Extract image filename and its writer id
            info = line.split()
            image_path, writer_id = info[0], info[1]

            # If its the first image for the writer, then the writer to the dictionary
            if writer_id not in self.writers:
                self.writers[writer_id] = []

            # Append image filename to its writer entry in the dictionary
            self.writers[writer_id].append(image_path)

        # Close meta data file
        meta.close()

        #
        # Remove any writer with less than 3 images
        #
        to_be_removed = []
        for writer in self.writers:
            if len(self.writers[writer]) < 3:
                to_be_removed.append(writer)
        for writer in to_be_removed:
            self.writers.pop(writer)

        # Get list of all writers ids
        self.writers_ids = list(self.writers.keys())

    def generate(self, path, n, m, k):
        """
        Generates test iterations.

        :param path:
        :param n: number of test iterations to generate.
        :param m: number of writers per test iteration.
        :param k: number of images per writer per test iteration.
        """

        # Delete old test iterations
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        # Open output file to write the expected results of the test iterations
        self.output_file = open(path + EXPECTED_RESULTS_FILENAME, 'w')

        # Generate the n test iterations
        for i in range(n):
            print('Generating test iteration', i, '...')
            test_path = path + format(i, '03d') + '/'
            self.generate_test(test_path, m, k)

        # Close output file
        self.output_file.close()

    def generate_test(self, path, m, k):
        used_writers = {}
        used_images = set()

        #
        # Pick m different random writers
        #
        for i in range(m):
            # Get random writer
            w = random.choice(self.writers_ids)
            while w in used_writers:
                w = random.choice(self.writers_ids)
            used_writers[w] = i

            # Get list of images written by this writer
            images = self.writers[w]

            #
            # Pick k different random images for the i-th writer
            #
            for j in range(k):
                # Get random image
                img = random.choice(images)
                while img in used_images:
                    img = random.choice(images)
                used_images.add(img)

                # Copy the image to its new path
                src_img = self.dataset_path + img + '.png'
                dst_img = path + str(i) + '/' + img + '.png'
                copy_file(src_img, dst_img)

        #
        # Pick random test image
        #

        # Pick random writer from the used ones
        w = random.choice(list(used_writers.keys()))
        images = self.writers[w]

        # Pick new random image of this writer
        img = random.choice(images)
        while img in used_images:
            img = random.choice(images)

        # Copy the image to its new path
        src_img = self.dataset_path + img + '.png'
        dst_img = path + img + '.png'
        copy_file(src_img, dst_img)

        # Write expected writer
        self.output_file.write(str(used_writers[w]) + '\n')
