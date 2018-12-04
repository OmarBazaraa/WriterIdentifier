import numpy as np
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as mlp


class Classifier:
    def __init__(self, mtype, x, y):
        # TODO modify these parameters
        if mtype == 'svm':
            self.model = svm.SVC(C=1.0)
        elif mtype == 'mlp':
            self.model = mlp(solver='sgd', activation='logistic', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)

        # Save the writers read used in case of not using the full dataset.
        self.train_writers = set(self.y_train)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self):
        y_pred = self.model.predict(self.x_test)

        correct_pred = np.sum(y_pred == self.y_test)

        # TODO be removed when trianing on full dataset.
        not_found_writers_count = 0
        for writer in self.y_test:
            if writer not in self.train_writers:
                not_found_writers_count += 1

        print(correct_pred, len(self.y_test), not_found_writers_count)

        return correct_pred / (len(self.y_test) - not_found_writers_count) * 100


class GMMModel:
    def __init__(self, writers_features):
        # Create a gmm model for each writer.

        # Save the writers read used in case of not using the full dataset.
        self.writers_models = {}
        self.writers_features = writers_features
        self.writers_probs = {}

    def get_writers_models(self):
        for writer_id, writer_features in self.writers_features.items():
            self.writers_models[writer_id] = GaussianMixture(n_components=len(writer_features),
                                                             covariance_type='diag', tol=0.001, reg_covar=1e-06,
                                                             max_iter=100, n_init=1,
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

    def evaluate(self):
        correct = []
        # Loop over all writers.
        for writer_id, writer_features in self.writers_features.items():
            max_prob = -1000
            pred_writer_id = -1
            # Predict each line whether it belongs to which writer.
            lines_probabilities = []
            idx = {}
            i = 0
            for key in self.writers_models.keys():
                idx[key] = i
                i += 1
                probabilities = self.writers_models[key].predict(np.reshape(writer_features, (len(writer_features), 7)))
                lines_probabilities.append(np.log2(probabilities))

            lines_probabilities = np.asarray(lines_probabilities).reshape(
                (len(lines_probabilities), len(lines_probabilities[0])))

            predictions = [idx[k] for k in np.argmax(lines_probabilities, axis=0)]
            correct.extend(predictions == writer_id)

        return np.mean(correct)
