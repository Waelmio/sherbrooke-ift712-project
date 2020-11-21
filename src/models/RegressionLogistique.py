from sklearn.linear_model import LogisticRegression

class Logistique():
    """A simple classifier interface"""

    def __init__(self):
        #solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
        #Beaucoup d'over-fitting
        self.model = LogisticRegression(penalty = 'none', solver = 'lbfgs')

    def fit(self, X, Y):
        self.model.fit(X,Y)

    def predict(self, X):
        """Need to be a list of, for each point,
        a list of the probability of that point to be in each classes"""
        return self.model.predict(X)
