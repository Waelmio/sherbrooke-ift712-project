from sklearn.linear_model import Ridge

class LinearRidge():
    """A simple classifier interface"""

    def __init__(self):
        self.model = Ridge(alpha = 0.01, normalize = True)

    def fit(self, X, Y):
        self.model.fit(X,Y)

    def predict(self, X):
        """Need to be a list of, for each point,
        a list of the probability of that point to be in each classes"""
        return self.model.predict(X)
