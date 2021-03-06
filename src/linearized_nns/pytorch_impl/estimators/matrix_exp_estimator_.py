import torch
import torch.nn as nn
import time
from linearized_nns.estimator import Estimator
from linearized_nns.pytorch_impl.matrix_exp import compute_exp_term
from linearized_nns.pytorch_impl.nns.utils import to_one_hot


class MatrixExpEstimator(Estimator):
    def __init__(self, model, num_classes, device, criterion=None, learning_rate=1., step=1024, momentum=0.,
                 reg_param=0., aug_grad=False):
        self.model = model
        self.device = device
        self.std_reciprocals = self.get_std_reciprocals()
        self.num_classes = num_classes

        self.criterion = self.default_criterion if (criterion is None) else criterion

        self.reg_param = reg_param

        w_elem = 0. * self.get_model_param()
        if aug_grad:
            w_elem = torch.cat([w_elem, w_elem])

        w = torch.stack([w_elem for _ in range(num_classes)])
        self.ws = w.detach()

        self.step = step
        self.momentum = momentum
        self.lr = learning_rate
        self.set_learning_rate(learning_rate)
        self.optimizer = torch.optim.SGD([self.ws], lr=1., momentum=self.momentum)

        self.last_pred = None
        self.last_pred_change = None

        self.aug_grad = aug_grad
        self.aug_c = None

    def clone(self):
        estimator = MatrixExpEstimator(model=self.model,
                                       num_classes=self.num_classes,
                                       device=self.device,
                                       criterion=self.criterion,
                                       learning_rate=self.lr,
                                       momentum=self.momentum,
                                       reg_param=self.reg_param,
                                       aug_grad=self.aug_grad)
        estimator.set_ws(self.ws)
        estimator.aug_c = self.aug_c
        return estimator

    def set_learning_rate(self, learning_rate):
        self.lr = learning_rate

    def get_learning_rate(self):
        return self.lr

    def set_ws(self, ws):
        self.ws = ws.clone().detach().to(self.device)
        self.optimizer = torch.optim.SGD([self.ws], lr=1., momentum=self.momentum)

    def get_ws(self):
        return self.ws.clone().detach()

    def fit_batch(self, X, y):
        self.zero_grad()
        start_time = time.time()

        y_pred = self.predict(X).detach()
        y_pred.requires_grad = True

        loss = self.criterion(y_pred, y)
        loss.backward()
        y_residual = y_pred.grad

        print(f"accuracy before fit {(y_pred.argmax(dim=1) == y).float().mean().item():.5f}, loss {loss.item():.5f}")

        y_diff = y_pred - to_one_hot(y, self.num_classes).to(self.device)
        scale = (y_diff * y_residual).sum() / (y_residual ** 2).sum()

        self.fit_residuals(X, y_residual * scale)

        print(f"fitting done. took {time.time() - start_time:.0f}s")
        print()

    def fit_residuals(self, X, y_residual):
        theta_0 = self.compute_theta_0(X)
        print(f"computing grads ...")

        with torch.no_grad():
            n = len(X)
            print(f"exponentiating kernel matrix ...")
            exp_term = - self.lr * compute_exp_term(- self.lr * theta_0, self.device)
            right_vector = torch.matmul(exp_term, y_residual)
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

        pred_change = torch.matmul(theta_0, right_vector)
        self.ws.grad = -torch.stack(ws)
        self.optimizer.step()

        return pred_change

    def get_model_param(self):
        return torch.cat([param.view(-1) for param in self.model.parameters()]).detach()

    def predict(self, X):
        def predict_one(x):
            with torch.no_grad():
                f0 = self.model.forward(x.unsqueeze(0)).view(-1)
            return (self.__grad(x) * self.ws).sum(dim=1) + f0.detach()

        return torch.stack([predict_one(x) for x in X]).detach()

    def default_criterion(self, prediction, y):
        y = to_one_hot(y, self.num_classes).to(self.device)
        return nn.MSELoss()(prediction, y)

    def zero_grad(self):
        self.ws.grad = None

    def grads(self, X):
        return torch.stack([self.__grad(x) for x in X]).detach()

    def __grad(self, X):
        return self.__grad_aug(X) if self.aug_grad else self.__grad_no_aug(X)

    def __grad_no_aug(self, x):
        self.model.zero_grad()
        pred = self.model.forward(x.unsqueeze(0))[:, 0]
        pred.backward()
        grads = []
        for param, std_reciprocal in zip(self.model.parameters(), self.std_reciprocals):
            cur_grad = param.grad
            grads.append(std_reciprocal * cur_grad.view(-1))
        grad = torch.cat(grads).detach()
        return grad

    def __grad_aug(self, x):
        grad = self.__grad_no_aug(x)
        if self.aug_c is None:
            self.aug_c = (torch.abs(grad[grad != 0])).mean()
            print(f"aug_c {self.aug_c}")

        non_linearity = 2. * self.aug_c * (grad > self.aug_c).float()
        return torch.cat([grad, non_linearity])

    def get_std_reciprocals(self):
        res = []
        for param in self.model.parameters():
            std = torch.sqrt((param.view(-1) ** 2).mean()).item()
            std = 1. if (abs(std) < 1e-6) else std
            res.append(1. / std)
        return res

    def compute_theta_0(self, X):
        n = X.size()[0]
        theta_0 = torch.zeros([n, n]).to(self.device)
        for li in range(0, n, self.step):
            ri = min(li + self.step, n)
            grads_i = self.grads(X[li:ri])

            for lj in range(0, ri, self.step):
                rj = min(lj + self.step, n)
                grads_j = self.grads(X[lj:rj])
                with torch.no_grad():
                    mat_prod = torch.matmul(grads_i, grads_j.T)
                    theta_0[li:ri, lj:rj] = mat_prod
                    theta_0[lj:rj, li:ri] = mat_prod.T
                del grads_j
            del grads_i
        return theta_0 + self.reg_param * torch.eye(n).to(self.device)

    def _fit_boosted(self, X, y, n_iter=10):
        n = len(X)
        index_size = int(n * 0.63)

        theta_0 = self.compute_theta_0(X)

        y_pred = self.predict(X).detach()
        right_vector = y_pred * 0

        for i in range(n_iter):

            print(f"i = {i}, loss : {self.criterion(y_pred, y):.5f}, exponentiating small kernel matrix ...")
            index = torch.randperm(n).to(self.device)[:index_size]

            cur_theta_0 = theta_0[index][:, index]

            y_pred = y_pred.clone().detach()
            y_pred.requires_grad = True
            loss = self.criterion(y_pred, y)
            loss.backward()
            y_residual = y_pred.grad.clone().detach()
            y_pred = y_pred.clone().detach()

            y_diff = y_pred - to_one_hot(y, self.num_classes).to(self.device)
            scale = (y_diff * y_residual).sum() / (y_residual ** 2).sum()
            y_residual *= scale

            exp_term = - self.lr * compute_exp_term(- self.lr * cur_theta_0, self.device)

            d_right_matrix = torch.zeros([n, n]).to(self.device)

            for row, ind in zip(exp_term, index):
                d_right_matrix[ind, index] = row

            pred_change = torch.matmul(theta_0, torch.matmul(d_right_matrix, y_residual))
            beta = self.find_beta(y_pred, pred_change, y)
            print(beta)

            right_vector += beta * torch.matmul(d_right_matrix, y_residual)
            y_pred += beta * pred_change

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

        pred_change = torch.matmul(theta_0, right_vector)
        self.ws.grad = -torch.stack(ws)
        self.optimizer.step()

        print(f"loss : {self.criterion(y_pred, y)}")

        return pred_change

    def find_beta(self, y_pred, pred_change, y, n_iter=1000):
        y_pred = y_pred.clone().detach()
        pred_change = pred_change.clone().detach()
        beta = torch.tensor(1.).to(self.device)
        beta.requires_grad = True
        learning_rate = 0.01
        for i in range(n_iter):
            loss = self.criterion(y_pred + beta * pred_change, y)
            loss.backward()
            beta.data -= learning_rate * beta.grad
            beta.grad = torch.tensor(0.).to(self.device)
        return max(0., beta.detach().item())
