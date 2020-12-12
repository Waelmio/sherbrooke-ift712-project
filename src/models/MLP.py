"""
MLP

Bougeard Yann 20137996
Wilmo Maël 20 138 003
"""

from models.classifierModel import Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from threading import Thread
import time
import os


os.environ['PYTHONWARNINGS'] = "ignore"


class MLP(Classifier):
    """A MLP classifier with k-fold cross-validation to search hyperparameters
       and threadings to increase speed."""

    def __init__(self):
        self.K_CV_NUM = 5
        self.model = MLPClassifier()

    def fit(self, X, Y):
        alpha_values = [0, 1e-5, 1e-4, 1e-3,
                        1e-2, 0.1, 0.2, 0.4, 0.8,
                        1, 2]
        layer_sizes = [100, 200, 300]
        hidden_layer_sizes = []

        # 1 hidden layer
        # each of a size in layer_sizes
        for i in range(0, len(layer_sizes)):
            hidden_layer_sizes.append((layer_sizes[i], ))

        param_grid = [
                      {
                        'activation': ['tanh', 'relu'],
                        'solver': ['lbfgs', 'adam'],
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'alpha': alpha_values
                      }
                     ]
        start_time = time.time()
        self.model = GridSearchCV(MLPClassifier(), param_grid,
                                  cv=self.K_CV_NUM, scoring='accuracy',
                                  n_jobs=-1)
        self.model.fit(X, Y)

        # print("Best parameters set found on development set in ",
        #       time.time() - start_time, "s:")
        # print(self.model.best_params_)

    def predict(self, X):
        """Need to be a list of the predicted class for each sample."""
        return self.model.predict(X)
