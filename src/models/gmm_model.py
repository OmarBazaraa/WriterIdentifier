import numpy as np
from sklearn.mixture import GaussianMixture


class GMMModel:
    COUNT_FEATURES = 8

    def __init__(self, writers_features):
        # Create a gmm model for each writer.
        self.writers_models = {}
        self.writers_features = writers_features

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

    def predict(self, features):
        predictions = []
        idx_id_dic = {}
        i = 0
        for writer_id, writer_model in self.writers_models.items():
            idx_id_dic[writer_id] = i
            i += 1
            predictions.append(writer_model.predict(features))

        return idx_id_dic[np.argmax(predictions)]

    def evaluate(self, t):
        classification = []
        # Loop over all writers.
        for writer_id, writer_features in t.items():
            for image_features in writer_features:
                # Predict each line whether it belongs to which writer.
                image_probabilities = []
                idx = {}
                i = 0
                for key in self.writers_models.keys():
                    idx[i] = key
                    i += 1
                    probabilities = self.writers_models[key].score(
                        np.reshape(image_features, (len(image_features), GMMModel.COUNT_FEATURES)))
                    image_probabilities.append(probabilities)
                prediction = idx[np.argmax(image_probabilities)]
                classification.append(prediction == writer_id)

        return np.mean(classification)
