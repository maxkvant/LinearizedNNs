from estimator import Estimator


class BaggingEstimator(Estimator):
    def __init__(self, estimator_constructor):
        self.estimator_constructor = estimator_constructor

        self.base_estimator = estimator_constructor()
        self.ws_sum = None
        self.items  = 0

    def fit(self, X, y):
        cur_estimator = self.estimator_constructor()
        cur_estimator.ws = cur_estimator.ws
        cur_estimator.fit(X, y)

        if self.ws_sum is None:
            self.ws_sum = cur_estimator.ws
        self.items += 1

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
