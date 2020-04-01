from estimator import Estimator
from pytorch_impl.estimators import MatrixExpEstimator


class BaggingEstimator(Estimator):
    def __init__(self, estimator_constructor):
        self.estimator_constructor = estimator_constructor

        self.base_estimator: MatrixExpEstimator = estimator_constructor()
        self.ws_sum = None
        self.items  = 0
        self.learning_rate = self.base_estimator.get_learning_rate()

    def fit(self, X, y):
        cur_estimator: MatrixExpEstimator = self.estimator_constructor()
        cur_estimator.set_ws(self.base_estimator.ws)
        cur_estimator.set_learning_rate(self.learning_rate)

        cur_estimator.fit(X, y)

        if self.ws_sum is None:
            self.ws_sum = cur_estimator.ws
        else:
            self.ws_sum += cur_estimator.ws.detach()
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
        self.ws_sum = None
        self.items  = 0

    def get_current_estimator(self) -> MatrixExpEstimator:
        if self.ws_sum is None:
            return self.base_estimator

        estimator: MatrixExpEstimator = self.estimator_constructor()
        estimator.set_ws(self.ws_sum / self.items)
        estimator.set_learning_rate(self.learning_rate)
        return estimator
