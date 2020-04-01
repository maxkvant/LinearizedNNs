import torch
import torch.nn as nn
import time
from estimator import Estimator
from pytorch_impl.matrix_exp import compute_exp_term
from pytorch_impl.nns.utils import to_one_hot


class MatrixExpEstimator(Estimator):
    def __init__(self, model, num_classes, device, criterion=None, learning_rate=1., step=1024, momentum=0.):
        self.model = model
        self.device = device
        self.lr = learning_rate
        self.num_classes = num_classes

        self.criterion = self.default_criterion if (criterion is None) else criterion

        w_elem = torch.cat([0 * param.view(-1) for param in self.model.parameters()])
        w = torch.stack([w_elem for _ in range(num_classes)])
        self.ws = w.detach()
        self.step = step

        self.optimizer = torch.optim.SGD([self.ws], lr=1., momentum=momentum)

    def set_learning_rate(self, learning_rate):
        self.lr = learning_rate

    def fit(self, X, y):
        self.zero_grad()
        start_time = time.time()

        y_pred = self.predict(X).detach()
        y_pred.requires_grad = True

        loss = self.criterion(y_pred, y)
        loss.backward()

        y_residual = y_pred.grad

        y_diff = y_pred - to_one_hot(y, self.num_classes)
        scale = (y_diff * y_residual).sum() / (y_residual ** 2).sum()

        print(f"scale {scale}")

        print(f"accuracy {(y_pred.argmax(dim=1) == y).float().mean().item():.5f}, loss {loss.item() / len(X):.5f}")

        theta_0 = self.compute_theta_0(X)
        print(f"computing grads ... {time.time() - start_time:.0f}s")

        with torch.no_grad():
            n = len(X)
            print(f"exponentiating kernel matrix ... {time.time() - start_time:.0f}s")
            exp_term = - self.lr * compute_exp_term(- self.lr * theta_0, self.device)
            right_vector = torch.matmul(exp_term, -y_residual * scale)
            del exp_term

        ws = [None for _ in range(self.num_classes)]
        for l in range(0, n, self.step):
            r = min(l + self.step, n)
            grads = self.grads(X[l:r])
            for i in range(self.num_classes):
                with torch.no_grad():
                    cur_w = torch.mv(grads.T, right_vector[l:r, i])
                    if ws[i] is None:
                        ws[i] = cur_w
                    else:
                        ws[i] = ws[i] + cur_w

        # TODO: optimize scale for gradient boosting

        ws = torch.stack(ws)
        self.ws.grad = torch.autograd.Variable(ws)
        self.optimizer.step()

    def predict(self, X):
        def predict_one(x):
            with torch.no_grad():
                f0 = self.model.forward(x).view(-1)
            return (self.__grad(x) * self.ws).sum(dim=1) + f0.detach()
        return torch.stack([predict_one(x) for x in X]).detach()

    def default_criterion(self, prediction, y):
        y = to_one_hot(y, self.num_classes).to(self.device)
        return nn.MSELoss()(prediction, y)

    def zero_grad(self):
        self.ws.grad = None

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

    def compute_theta_0(self, X):
        n = X.size()[0]

        theta_0 = torch.zeros([n,n]).to(self.device)
        for li in range(0, n, self.step):
            ri = min(li + self.step, n)
            grads_i = self.grads(X[li:ri])

            for lj in range(0, n, self.step):
                rj = min(lj + self.step, n)
                grads_j = self.grads(X[lj:rj])
                with torch.no_grad():
                    theta_0[li:ri, lj:rj] = torch.matmul(grads_i, grads_j.T)
                del grads_j
            del grads_i
        return theta_0
