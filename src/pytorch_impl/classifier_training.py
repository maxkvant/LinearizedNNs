import time

import numpy as np
import matplotlib.pyplot as plt

from training import Training


class ClassifierTraining(Training):
    def __init__(self, estimator, device):
        self.estimator = estimator
        self.device = device

        self.test_accuracies  = []
        self.train_accuracies = []

    def train(self, train_loader, test_loader, num_epochs=10, learning_rate=0.01):
        self.estimator.set_learning_rate(learning_rate)
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"epoch {epoch}/{num_epochs}, {time.time() - start_time:.0f}s since start")

            for batch_id, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                self.estimator.fit(X, y)

            train_accuracy = self.get_accuracy(train_loader, limit=2000)
            test_accuracy  = self.get_accuracy(test_loader,  limit=2000)
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)

            self.plot_performance()

        print(f"training took {time.time() - start_time:.0f}s")
        print(f"test_accuracy {self.get_accuracy(test_loader):.3f}")

    def get_accuracy(self, dataloader, limit=None):
        cur_accuracies = []

        cur_size = 0
        for batch_id, (X, y) in enumerate(dataloader):
            cur_size += len(X)

            X, y = X.to(self.device), y.to(self.device)

            output = self.estimator.predict(X)
            y_pred = output.argmax(dim=1)
            cur_accuracies.append((y == y_pred).double().mean().item())

            if limit is not None and cur_size >= limit:
                break
        return np.average(cur_accuracies)

    def plot_performance(self):
        l = len(self.test_accuracies)
        plt.plot(np.arange(l), self.train_accuracies, label='train')
        plt.plot(np.arange(l), self.test_accuracies,  label='test')
        plt.legend()
        plt.show()
