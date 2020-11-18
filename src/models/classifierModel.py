class Classifier():
    """A simple classifier interface"""

    def __init__(self):
        pass

    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        """Need to be a list of, for each point,
        a list of the probability of that point to be in each classes"""
        raise NotImplementedError
