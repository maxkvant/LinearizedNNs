import torch

from estimator import Estimator


class SgdEstimator(Estimator):
    def __init__(self, model, criterion, learning_rate):
        self.model = model
        self.criterion = criterion

        self.optimizer = None
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, learning_rate):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def fit(self, X, y):
        self.model.zero_grad()

        output = self.model.forward(X)
        loss   = self.criterion(output, y)
        loss.backward()

        self.optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self.model.forward(X)
