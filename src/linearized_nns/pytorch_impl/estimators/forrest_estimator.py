import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from linearized_nns.estimator import Estimator


class ForrestEstimator(Estimator):
    def __init__(self, model, input_channels, device, sample_size=64):
        self.model = model

        self.learning_rate = None
        self.input_channels = input_channels
        self.classifier = RandomForestClassifier(n_estimators=500, max_depth=3)
        self.device = device
        self.sample_size = sample_size

    def fit_batch(self, X, y):
        train_X = []
        train_y = []
        for i, (x, cur_y) in enumerate(zip(X, y)):
            cur_X = self.to_train_subset(x)
            train_X.extend(cur_X)

            cur_y = cur_y.cpu().numpy()
            cur_y = np.repeat(cur_y, len(cur_X))
            train_y.extend(cur_y)

        train_X = np.asarray(train_X)
        print(f"train_size {train_X.shape}")
        self.classifier.fit(train_X, train_y)

    def predict(self, X):
        y_pred = []
        for x in X:
            cur_X = self.to_train_subset(x)
            alpha = 0.01
            log_proba = np.log(alpha + (1. - alpha) * self.classifier.predict_proba(cur_X))
            log_proba = np.average(log_proba, axis=0)
            y_pred.append(log_proba)
        return torch.tensor(y_pred, dtype=torch.float32).to(self.device)

    def to_train_subset(self, x):
        self.model.zero_grad()
        output = self.model.forward(x.unsqueeze(0))
        output.backward()

        feature_values = []

        def to_feature(grad_part, param_part):
            grad_part  =  grad_part.view(-1)
            n = len(grad_part)
            copies = self.sample_size // n + 1
            index  = torch.cat([torch.randperm(n) for _ in range(copies)])
            index = index[:self.sample_size]
            return grad_part[index].cpu().numpy()

        for param in self.model.parameters():
            if len(param.shape) != 4:
                continue
            grad = param.grad.clone().detach()
            param = param.clone().detach()
            assert param.shape == grad.shape

            _, input_channels, w, h = param.shape
            if input_channels == self.input_channels:
                for channel in range(input_channels):
                    for i in range(w):
                        for j in range(h):
                            cur_grad =   grad[:, channel, i, j]
                            cur_param = param[:, channel, i, j]
                            feature_values.append(to_feature(cur_grad, cur_param))
            else:
                for i in range(w):
                    for j in range(h):
                        cur_grad = grad[:, :, i, j]
                        cur_param = param[:, :, i, j]
                        feature_values.append(to_feature(cur_grad, cur_param))
        feature_values = np.asarray(feature_values)
        return feature_values.T

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.learning_rate
