from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from threading import Thread
import time


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


class MLP():
    """A MLP classifier with k-fold cross-validation to search hyperparameters
       and threadings to increase speed."""

    def __init__(self):
        self.K_CV_NUM = 5
        self.model = MLPClassifier()

    def compute_error(self, Y_pred, Y):
        """Return the percentage of data that wasn't in the good class"""
        err = 0
        for i in range(0, len(Y_pred)):
            if Y_pred[i] != Y[i]:
                err += 1
        return err / len(Y_pred) * 100

    def fit_one(self, X, Y, hidden_layer_sizes, alpha):
        err = 0

        sss = StratifiedShuffleSplit(n_splits=self.K_CV_NUM,
                                     test_size=0.2)

        # k-fold cross validation with k = K_CV_NUM
        for train_index, valid_index in sss.split(X, Y):
            X_train, X_valid = \
                X[train_index], X[valid_index]
            y_train, y_valid = Y[train_index], Y[valid_index]
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                  alpha=alpha, max_iter=500)
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            err += self.compute_error(pred, y_valid)

        err /= self.K_CV_NUM

        print("        Alpha: ", alpha, ", Hidden: ", hidden_layer_sizes)

        return err, hidden_layer_sizes, alpha

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
            # for j in range(0, len(layer_sizes)):
            #     hidden_layer_sizes.append((layer_sizes[i], layer_sizes[j],))
            #     for k in range(0, len(layer_sizes)):
            #         hidden_layer_sizes.append((layer_sizes[i],
            #                                    layer_sizes[j], layer_sizes[k],)
            #                                   )
        # print(hidden_layer_sizes)

        # m_error = 101  # Error in percent
        # m_alpha = 0
        # m_hidden = None

        # threads = []

        # for hidden_layers in hidden_layer_sizes:
        #     for alpha in alpha_values:
        #         the_thread = Threader(target=self.fit_one,
        #                               args=(X, Y, hidden_layers, alpha))
        #         threads.append(the_thread)
        #         the_thread.start()

        # for thread in threads:
        #     err, kern, C = thread.join()
        #     if err < m_error:
        #         print("        Layers: ", m_hidden, " -> ", hidden_layers,
        #               ", alpha: ", m_alpha, " -> ", alpha,
        #               ", Error: ", m_error, " -> ", err)
        #         m_hidden = hidden_layers
        #         m_alpha = alpha
        #         m_error = err

        # self.model = MLPClassifier(hidden_layer_sizes=m_hidden,
        #                            alpha=m_alpha)
        # self.model.fit(X, Y)

        param_grid = [
                      {
                        'activation': ['tanh'],
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

        print("Best parameters set found on development set in ",
              time.time() - start_time, "s:")
        print(self.model.best_params_)

    def predict(self, X):
        """Need to be a list of the predicted class for each sample."""
        return self.model.predict(X)
