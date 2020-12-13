"""
Ridge

Bougeard Yann 20137996
Wilmo Maël 20 138 003
"""

from models.classifierModel import Classifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from threading import Thread


class Threader(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        super(Threader, self).__init__(group, target, name, args, kwargs)
        self._return = None
        self._args = args
        self._target = target

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args)

    def join(self):
        Thread.join(self)
        return self._return

class LinearRidge(Classifier):
    """A simple classifier interface"""

    def __init__(self):
        self.K_CV_NUM = 5
        self.model = RidgeClassifier(alpha=0.01, solver='auto', normalize=True)

    def compute_error(self, Y_pred, Y):
        """Return the percentage of data that wasn't in the good class"""
        err = 0
        for i in range(0, len(Y_pred)):
            if Y_pred[i] != Y[i]:
                err += 1
        return err / len(Y_pred) * 100

    def search_parameters(self, X, Y, alpha):
        err = 0

        sss = StratifiedShuffleSplit(n_splits=self.K_CV_NUM,
                                     test_size=0.2)

        # k-fold cross validation with k = K_CV_NUM
        for train_index, valid_index in sss.split(X, Y):
            X_train, X_valid = \
                X[train_index], X[valid_index]
            y_train, y_valid = Y[train_index], Y[valid_index]
            model = RidgeClassifier(alpha=alpha, 
                                    solver='auto')
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            err += self.compute_error(pred, y_valid)

        err /= self.K_CV_NUM

        return err, alpha

    def fit(self, X, Y):
        alphas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        m_alpha = 0
        m_error = 101
        threads = []

        for alpha in alphas:
            the_thread = Threader(target=self.search_parameters,
                                    args=(X, Y, alpha))
            threads.append(the_thread)
            the_thread.start()

        for thread in threads:
            err, alpha = thread.join()
            if err < m_error:
                m_alpha = alpha
                m_error = err

        self.model = RidgeClassifier(alpha=m_alpha,
                                solver='auto')
        self.model.fit(X, Y)

    def predict(self, X):
        """Need to be a list of, for each point,
        a list of the probability of that point to be in each classes"""
        return self.model.predict(X)
