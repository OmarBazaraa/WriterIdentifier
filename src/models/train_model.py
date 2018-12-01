import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


class SVMClassifier:
    def __init__(self, x, y):
        # TODO modify these parameters
        self.model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                             decision_function_shape='ovr', degree=3, kernel='rbf',
                             max_iter=-1, probability=False, random_state=None, shrinking=True,
                             tol=0.001, verbose=False)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self):
        y_pred = self.model.predict(self.x_test)

        correct_pred = np.sum(y_pred == self.y_test)

        return correct_pred / len(self.y_test)
