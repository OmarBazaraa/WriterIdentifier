import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier as mlp


class Classifier:
    def __init__(self, mtype, x, y, x_test=None, y_test=None):
        # TODO modify these parameters
        if mtype == 'svm':
            self.model = svm.SVC(C=1.0, gamma='scale')
        elif mtype == 'mlp':
            self.model = mlp(solver='sgd', activation='logistic', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)

        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)
        # Make it take only one image for each of the 2 writers writer
        self.x_train, self.x_test, self.y_train, self.y_test = x, x_test, y, y_test

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

        # print(correct_pred, len(self.y_test), not_found_writers_count)

        return correct_pred / (len(self.y_test) - not_found_writers_count) * 100
