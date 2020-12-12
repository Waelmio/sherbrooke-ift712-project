from models.classifierModel import Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
"""
LDA

Bougeard Yann 20137996
Wilmo Maël 20 138 003
"""

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


class LDA(Classifier):
    """A LDA classifier with k-fold cross-validation to search hyperparameters
       and threadings to increase speed."""

    def __init__(self):
        self.K_CV_NUM = 8
        self.model = LinearDiscriminantAnalysis()

    def compute_error(self, Y_pred, Y):
        """Return the percentage of data that wasn't in the good class"""
        err = 0
        for i in range(0, len(Y_pred)):
            if Y_pred[i] != Y[i]:
                err += 1
        return err / len(Y_pred) * 100

    def fit_one(self, X, Y, solv, shrink, tol):
        err = 0

        sss = StratifiedShuffleSplit(n_splits=self.K_CV_NUM,
                                     test_size=0.2)

        # k-fold cross validation with k = K_CV_NUM
        for train_index, valid_index in sss.split(X, Y):
            X_train, X_valid = \
                X[train_index], X[valid_index]
            y_train, y_valid = Y[train_index], Y[valid_index]
            model = LinearDiscriminantAnalysis(solver=solv, shrinkage=shrink,
                                               tol=tol)
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            err += self.compute_error(pred, y_valid)

        err /= self.K_CV_NUM

        return err, solv, shrink, tol

    def fit(self, X, Y):
        tol_values = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        solvers = ["svd", "lsqr", "eigen"]
        shrinkers = [None, "auto"]

        m_error = 101  # Error in percent
        m_solv = solvers[0]
        m_shrinker = None
        m_tol = 1e-4

        threads = []

        for tol in tol_values:
            for solv in solvers:
                for shrink in shrinkers:
                    # Jump this not implemented combination
                    if shrink == "auto" and solv == "svd":
                        continue
                    the_thread = Threader(target=self.fit_one,
                                          args=(X, Y, solv, shrink, tol))
                    threads.append(the_thread)
                    the_thread.start()

        for thread in threads:
            err, solv, shrink, tol = thread.join()
            if err < m_error:
                # print("        Solver: ", m_solv, " -> ", solv,
                #       ", Shrink: ", m_shrinker, " -> ", shrink,
                #       ", Tol: ", m_tol, " -> ", tol,
                #       ", Error: ", m_error, " -> ", err)
                m_tol = tol
                m_solv = solv
                m_shrinker = shrink
                m_error = err

        self.model = LinearDiscriminantAnalysis(solver=m_solv,
                                                shrinkage=m_shrinker,
                                                tol=m_tol)
        self.model.fit(X, Y)

    def predict(self, X):
        """Need to be a list of the predicted class for each sample."""
        return self.model.predict(X)
