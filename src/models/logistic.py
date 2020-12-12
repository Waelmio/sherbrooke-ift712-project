from sklearn.linear_model import LogisticRegression
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


class Logistic():

    def __init__(self):
        # solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
        # , default=’lbfgs’
        self.K_CV_NUM = 5
        self.model = LogisticRegression(penalty='none', solver='lbfgs')

    def compute_error(self, Y_pred, Y):
        """Return the percentage of data that wasn't in the good class"""
        err = 0
        for i in range(0, len(Y_pred)):
            if Y_pred[i] != Y[i]:
                err += 1
        return err / len(Y_pred) * 100

    def search_parameters(self, X, Y, solver):
        err = 0

        sss = StratifiedShuffleSplit(n_splits=self.K_CV_NUM,
                                     test_size=0.2)

        # k-fold cross validation with k = K_CV_NUM
        for train_index, valid_index in sss.split(X, Y):
            X_train, X_valid = \
                X[train_index], X[valid_index]
            y_train, y_valid = Y[train_index], Y[valid_index]
            if (solver == 'liblinear'):
                model = LogisticRegression(penalty='l1', solver=solver)
            else:
                model = LogisticRegression(penalty='none', solver=solver)
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            err += self.compute_error(pred, y_valid)

        err /= self.K_CV_NUM

        return err, solver

    def fit(self, X, Y):
        m_error = 101
        m_solver = None
        threads = []
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

        for solver in solvers:
            the_thread = Threader(target=self.search_parameters,
                                  args=(X, Y, solver))
            threads.append(the_thread)
            the_thread.start()

        for thread in threads:
            err, solver = thread.join()
            if err < m_error:
                m_solver = solver
                m_error = err

        print(m_solver)
        if (m_solver == 'liblinear'):
            self.model = LogisticRegression(penalty='l1', solver=m_solver)
        else:
            self.model = LogisticRegression(penalty='none', solver=m_solver)
        self.model.fit(X, Y)

    def predict(self, X):
        """Need to be a list of, for each point,
        a list of the probability of that point to be in each classes"""
        return self.model.predict(X)
