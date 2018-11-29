class FeatureExtractor:
    @staticmethod
    def extract_features(gray_img, bin_img, gray_lines, bin_lines):
        features = []

        features.append(FeatureExtractor.horizontal_run_length(gray_lines))

        return features

    @staticmethod
    def horizontal_run_length(gray_lines):
        """
        Get the run length feature given gray lines.
        :return: a number representing the horizontal run length.
        """
        ret = 0.0

        for line in gray_lines:
            dummy = 1

        return ret
