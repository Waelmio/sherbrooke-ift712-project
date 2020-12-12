"""
SVM

Bougeard Yann 20137996
Wilmo Maël 20 138 003
"""

from sklearn.svm import SVC
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


class SVM():
    """A SVM classifier with k-fold cross-validation to search hyperparameters
       and threadings to increase speed."""

    def __init__(self):
        self.K_CV_NUM = 8
        self.model = SVC(kernel="rbf", C=0.025)

    def compute_error(self, Y_pred, Y):
        """Return the percentage of data that wasn't in the good class"""
        err = 0
        for i in range(0, len(Y_pred)):
            if Y_pred[i] != Y[i]:
                err += 1
        return err / len(Y_pred) * 100

    def fit_one(self, X, Y, kern, C):
        err = 0

        sss = StratifiedShuffleSplit(n_splits=self.K_CV_NUM,
                                     test_size=0.2)

        # k-fold cross validation with k = K_CV_NUM
        for train_index, valid_index in sss.split(X, Y):
            X_train, X_valid = \
                X[train_index], X[valid_index]
            y_train, y_valid = Y[train_index], Y[valid_index]
            model = SVC(kernel=kern, C=C)
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            err += self.compute_error(pred, y_valid)

        err /= self.K_CV_NUM

        return err, kern, C

    def fit(self, X, Y):
        C_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                    1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                    1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        kernels = ["linear", "poly", "rbf", "sigmoid"]

        m_error = 101  # Error in percent
        m_C = 0
        m_kern = None

        threads = []

        for kern in kernels:
            for C in C_values:
                the_thread = Threader(target=self.fit_one,
                                      args=(X, Y, kern, C))
                threads.append(the_thread)
                the_thread.start()

        for thread in threads:
            err, kern, C = thread.join()
            if err < m_error:
                # print("        Kernel: ", m_kern, " -> ", kern, ", C: ", m_C,
                #       " -> ", C, ", Error: ", m_error, " -> ", err)
                m_C = C
                m_kern = kern
                m_error = err

        self.model = SVC(kernel=m_kern, C=m_C)
        self.model.fit(X, Y)

    def predict(self, X):
        """Need to be a list of the predicted class for each sample."""
        return self.model.predict(X)
