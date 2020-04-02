import torch

from estimator import Estimator
from pytorch_impl.estimators import MatrixExpEstimator
from pytorch_impl.nns.utils import to_one_hot


class GradientBoostingEstimator(Estimator):
    def __init__(self, estimator_constructor, num_classes, criterion, device):
        self.estimator_constructor = estimator_constructor
        self.num_classes = num_classes
        self.criterion = criterion
        self.device = device

        self.base_estimator: MatrixExpEstimator = estimator_constructor()

        self.ws_change_sum = None
        self.ys = []
        self.y_prediction_changes = []
        self.y_predictions_before = []

        self.learning_rate = self.base_estimator.get_learning_rate()

    def fit_batch(self, X, y):
        cur_estimator = self.new_partial_estimator()

        y_pred = self.predict(X).detach()
        y_residual = self.find_y_residual(y_pred, y)

        pred_change = cur_estimator._fit_batch(X, y_residual)
        cur_ws_change = (cur_estimator.ws - self.base_estimator.ws).detach()

        self.ys.append(y)
        self.y_predictions_before.append(y_pred)
        self.y_prediction_changes.append(pred_change)
        self.ws_change_sum = cur_ws_change if (self.ws_change_sum is None) else (self.ws_change_sum + cur_ws_change)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.get_learning_rate()

    def predict(self, X):
        estimator = self.get_current_estimator()
        return estimator.predict(X)

    def epoch_callback(self):
        self.base_estimator = self.get_current_estimator()

        self.ws_change_sum = None
        self.ys = []
        self.y_prediction_changes = []
        self.y_predictions_before = []

    def find_y_residual(self, y_pred, y):
        y_pred.requires_grad = True
        loss = self.criterion(y_pred, y)
        loss.backward()

        y_residual = y_pred.grad
        y_diff = y_pred - to_one_hot(y, self.num_classes).to(self.device)
        scale = (y_diff * y_residual).sum() / (y_residual ** 2).sum()
        return scale * y_residual

    def get_current_estimator(self) -> MatrixExpEstimator:
        if self.ws_change_sum is None:
            return self.base_estimator

        estimator: MatrixExpEstimator = self.estimator_constructor()
        estimator.set_learning_rate(self.learning_rate)

        l = len(self.ys)
        beta = self.find_beta(torch.cat(self.y_predictions_before),
                              torch.cat(self.y_prediction_changes),
                              torch.cat(self.ys))
        print(f"beta {beta}")

        new_ws = self.base_estimator.ws + beta * self.ws_change_sum / l
        estimator.set_ws(new_ws)
        return estimator

    def new_partial_estimator(self):
        estimator: MatrixExpEstimator = self.estimator_constructor()
        estimator.set_ws(self.base_estimator.ws)
        estimator.set_learning_rate(self.learning_rate)
        return estimator

    def find_beta(self, y_pred, pred_change, y, n_iter=500):
        beta = torch.tensor(1.).to(self.device)
        beta.requires_grad = True
        learning_rate = 0.01
        for i in range(n_iter):
            loss = self.criterion(y_pred + beta * pred_change, y)
            loss.backward()
            beta.data -= learning_rate * beta.grad
            beta.grad = torch.tensor(0.).to(self.device)
        return max(0., beta.detach().item())

