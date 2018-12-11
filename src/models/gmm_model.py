import numpy as np
from sklearn.mixture import GaussianMixture


class GMMModel:
    COUNT_FEATURES = 8

    def __init__(self, writers_features):
        # Create a gmm model for each writer.

        # Save the writers read used in case of not using the full dataset.
        self.writers_models = {}
        self.writers_features = writers_features
        self.writers_probs = {}

    def get_writers_models(self):
        for writer_id, writer_features in self.writers_features.items():
            self.writers_models[writer_id] = GaussianMixture(n_components=2,
                                                             covariance_type='diag', tol=0.00001, reg_covar=1e-06,
                                                             max_iter=10000, n_init=1,
                                                             init_params='kmeans', weights_init=None, means_init=None,
                                                             precisions_init=None,
                                                             random_state=None, warm_start=False, verbose=0,
                                                             verbose_interval=10)

            self.writers_models[writer_id].fit(writer_features)

        return self.writers_models

    def predict(self, x):
        self.writers_probs = {}
        for writer_id, writer_model in self.writers_models.items():
            # Predict and save the probability.
            self.writers_probs[writer_id] = writer_model.predict(x)

        return self.writers_probs

    def evaluate(self, t):
        correct = []
        # Loop over all writers.
        for writer_id, writer_features in t.items():
            for image_features in writer_features:
                # Predict each line whether it belongs to which writer.
                lines_probabilities = []
                idx = {}
                i = 0
                for key in self.writers_models.keys():
                    idx[i] = key
                    i += 1
                    probabilities = self.writers_models[key].score(
                        np.reshape(image_features, (len(image_features), GMMModel.COUNT_FEATURES)))
                    lines_probabilities.append(probabilities)
                predictions = idx[np.argmax(lines_probabilities)]
                correct.append(predictions == writer_id)

        return np.mean(correct)
