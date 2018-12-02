import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as mlp


class Classifier:
    def __init__(self,type, x, y):
        # TODO modify these parameters
        if type == 'svm':
            self.model = svm.SVC(C=1.0)
        elif type == 'mlp':
            self.model = mlp(solver='sgd', activation='tanh', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1,
                             max_iter=1000)

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