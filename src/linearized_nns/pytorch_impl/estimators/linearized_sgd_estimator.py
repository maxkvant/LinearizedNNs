import torch
from estimator import Estimator


class LinearizedSgdEstimator(Estimator):
    def __init__(self, model, num_classes, criterion, learning_rate=5e-3):
        self.model = model
        self.num_classes = num_classes
        self.criterion = criterion

        w_elem = torch.cat([0 * param.view(-1) for param in self.model.parameters()])
        w = torch.stack([w_elem for _ in range(num_classes)]).T
        self.w = w.detach().requires_grad_(True)

        self.optimizer = None
        self.learning_rate = None
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

    def get_learning_rate(self):
        return self.learning_rate

    def predict(self, X):
        return self.forward(X)

    def fit_batch(self, X, y):
        self.zero_grad()

        output = self.forward(X)
        loss = self.criterion(output, y)
        loss.backward()

        self.optimizer.step()

    def forward(self, X):
        grads = self.grads(X)
        with torch.no_grad():
            f0 = self.model.forward(X)
        return torch.matmul(grads, self.w) + f0.detach()

    def zero_grad(self):
        self.w.grad = None

    def parameters(self):
        return [self.w]

    def grads(self, X):
        return torch.stack([self.__grad(x) for x in X]).detach()

    def __grad(self, x):
        self.model.zero_grad()
        pred = self.model.forward(x.unsqueeze(0))[:,0]
        pred.backward()
        grads = []
        for param in self.model.parameters():
            cur_grad = param.grad
            grads.append(cur_grad.view(-1))
        grad = torch.cat(grads).detach()
        return grad
