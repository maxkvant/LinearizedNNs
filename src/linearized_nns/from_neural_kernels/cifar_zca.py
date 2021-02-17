import numpy as np
import torchvision.datasets as datasets
from .utils import to_zca

"""
Based on part of the following code https://github.com/modestyachts/neural_kernels_code/blob/master/cifar10/run_cnn.py

"""

def get_cifar_zca():
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    X_train = np.asarray(trainset.data).astype(np.float64)
    y_train = np.asarray(trainset.targets)
    X_test  = np.asarray(testset.data).astype(np.float64)
    y_test  = np.asarray(testset.targets)

    (X_train, X_test), global_ZCA = to_zca(X_train, X_test)

    X_train = np.transpose(X_train, (0,3,1,2))
    X_test  = np.transpose(X_test,  (0,3,1,2))

    return X_train, y_train, X_test, y_test
