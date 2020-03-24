import torch
import time
from estimator import Estimator
from pytorch_impl.matrix_exp import compute_exp_term
from pytorch_impl.nns.utils import to_one_hot


class MatrixExpEstimator(Estimator):
    def __init__(self, model, num_classes, device, learning_rate, step=1000):
        self.model = model
        self.device = device
        self.lr = learning_rate
        self.num_classes = num_classes
        w_elem = torch.cat([0 * param.view(-1) for param in self.model.parameters()])
        w = torch.stack([w_elem for _ in range(num_classes)])
        self.ws = w.detach()
        self.step = step

    def set_learning_rate(self, learning_rate):
        self.lr = learning_rate

    def fit(self, X, y):
        start_time = time.time()

        y          = to_one_hot(y, num_classes=self.num_classes).to(self.device)
        y_residual = y - self.predict(X)

        theta_0 = self.compute_theta_0(X)
        print(f"computing grads ... {time.time() - start_time:.0f}s")

        with torch.no_grad():
            n = len(X)
            print(f"exponentiating kernel matrix ... {time.time() - start_time:.0f}s")
            exp_term = - self.lr * compute_exp_term(- self.lr * theta_0, self.device)
            right_vector = torch.matmul(exp_term, (-y_residual).double())
            del exp_term

        ws = [None for _ in range(self.num_classes)]
        for l in range(0, n, self.step):
            r = min(l + self.step, n)
            grads = self.grads(X[l:r]).double()
            for i in range(self.num_classes):
                with torch.no_grad():
                    cur_w = torch.mv(grads.T.double(), right_vector[l:r, i])
                    if ws[i] is None:
                        ws[i] = cur_w
                    else:
                        ws[i] = ws[i] + cur_w
        self.ws += torch.stack(ws)

    def predict(self, X):
        def predict_one(x):
            return (self.__grad(x).double() * self.ws).sum(dim=1)
        return torch.stack([predict_one(x) for x in X]).detach()

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

    def compute_theta_0(self, X):
        n = X.size()[0]

        Theta_0 = torch.zeros([n,n]).double().to(self.device)
        for li in range(0, n, self.step):
            ri = min(li + self.step, n)
            grads_i = self.grads(X[li:ri]).double()

            for lj in range(0, n, self.step):
                rj = min(lj + self.step, n)
                grads_j = self.grads(X[lj:rj]).double()
                with torch.no_grad():
                    Theta_0[li:ri, lj:rj] = torch.matmul(grads_i, grads_j.T)
                del grads_j
            del grads_i
        return Theta_0
