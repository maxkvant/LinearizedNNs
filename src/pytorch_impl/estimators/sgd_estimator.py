import torch
import torch.nn as nn
from estimator import Estimator


class SgdEstimator(Estimator):
    def __init__(self, model, num_classes, learning_rate=5e-3):
        self.model = model
        w_elem = torch.cat([0 * param.view(-1) for param in self.model.parameters()])
        w = torch.stack([w_elem for _ in range(num_classes)]).T
        self.w = w.detach().requires_grad_(True)

        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    def predict(self, X):
        return self.forward(X)

    def fit(self, X, y):
        self.zero_grad()

        output = self.forward(X)
        criterion = nn.MSELoss()
        loss = criterion(output, y)
        loss.backward()

        self.optimizer.step()

    def forward(self, X):
        grads = self.grads(X)
        return torch.matmul(grads, self.w)

    def zero_grad(self):
        self.w.grad = None

    def parameters(self):
        return [self.w]

    def grads(self, X):
        return torch.stack([self.__grad(x) for x in X]).detach()

    def __grad(self, x):
        self.model.zero_grad()
        pred = self.model.forward(x.unsqueeze(0))
        pred.backward()
        grads = []
        for param in self.model.parameters():
            cur_grad = param.grad
            grads.append(cur_grad.view(-1))
        grad = torch.cat(grads).detach()
        return grad
