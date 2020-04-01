from estimator import Estimator


class BaggingEstimator(Estimator):
    def __init__(self, estimator_constructor):
        self.estimator_constructor = estimator_constructor

        self.base_estimator: Estimator = estimator_constructor()
        self.ws_sum = None
        self.items  = 0
        self.learning_rate = self.base_estimator.get_learning_rate()

    def fit(self, X, y):
        cur_estimator: Estimator = self.estimator_constructor()
        cur_estimator.set_learning_rate(self.learning_rate)

        cur_estimator.ws = self.base_estimator.ws
        cur_estimator.fit(X, y)

        if self.ws_sum is None:
            self.ws_sum = cur_estimator.ws
        self.items += 1

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.get_learning_rate()

    def predict(self, X):
        estimator = self.get_current_estimator()
        return estimator.predict(X)

    def epoch_callback(self):
        self.base_estimator = self.get_current_estimator()

    def get_current_estimator(self):
        if self.ws_sum is None:
            return self.base_estimator

        estimator = self.estimator_constructor
        estimator.ws = estimator.ws / self.items
        return estimator
