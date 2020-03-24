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
        self.ws = None
        self.step = step

    def set_learning_rate(self, learning_rate):
        self.lr = learning_rate

    def fit(self, X, y):
        start_time = time.time()

        y = to_one_hot(y, num_classes=self.num_classes)

        theta_0 = self.compute_theta_0(X)
        print(f"computing grads ... {time.time() - start_time:.0f}s since start")

        with torch.no_grad():
            n = len(X)
            print(f"exponentiating kernel matrix ... {time.time() - start_time:.0f}s since start")
            exp_term = - self.lr * compute_exp_term(- self.lr * theta_0, self.device)
            right_vector = torch.matmul(exp_term, (-y).double())
            del exp_term

        ws = [None for _ in range(self.num_classes)]
        for l in range(0, n, self.step):
            r = min(l + self.step, n)
            grads = self.grads_logit(X[l:r]).double()
            for i in range(self.num_classes):
                with torch.no_grad():
                    cur_w = torch.mv(grads.T.double(), right_vector[l:r, i])
                    if ws[i] is None:
                        ws[i] = cur_w
                    else:
                        ws[i] = ws[i] + cur_w
        self.ws = torch.stack(ws)

    def predict(self, X):
        assert self.ws is not None

        def predict_one(x):
            return (self.grads_x(x).double() * self.ws).sum(dim=1)
        return torch.stack([predict_one(x) for x in X])

    def one_grad(self, pred_elem):
        self.model.zero_grad()
        pred_elem.backward(retain_graph=True)
        grads = []
        for param in self.model.parameters():
            cur_grad = param.grad
            grads.append(cur_grad.view(-1))
        grad = torch.cat(grads).view(-1).detach()
        return grad

    def grads_x(self, x):
        pred = self.model.forward(x.unsqueeze(0))
        return torch.stack([self.one_grad(elem) for elem in pred[0]])

    def grads(self, X):
        return torch.cat([self.grads_x(x) for x in X])

    def grads_logit(self, X):
        pred = self.model.forward(X)[:,0]
        return torch.stack([self.one_grad(elem) for elem in pred]).detach()

    def compute_theta_0(self, X):
        n = X.size()[0]

        Theta_0 = torch.zeros([n,n]).double().to(self.device)
        for li in range(0, n, self.step):
            ri = min(li + self.step, n)
            grads_i = self.grads_logit(X[li:ri]).double()

            for lj in range(0, n, self.step):
                rj = min(lj + self.step, n)
                grads_j = self.grads_logit(X[lj:rj]).double()
                with torch.no_grad():
                    Theta_0[li:ri, lj:rj] = torch.matmul(grads_i, grads_j.T)
                del grads_j
            del grads_i
        return Theta_0
