from sklearn.svm import SVC


class SVM():
    """A simple classifier interface"""

    def __init__(self):
        self.model = SVC(kernel="rbf", C=0.025)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        """Need to be a list of the predicted class for each sample."""
        return self.model.predict(X)
