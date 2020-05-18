import torch
import numpy as np

from .transforms import Cutout, RandomCrop

"""
Based on part of the following code https://github.com/modestyachts/neural_kernels_code/blob/master/cifar10/run_cnn.py

"""


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, data, targets, transform=None):
        assert data.size(0) == targets.size(0)
        self.data = data
        self.targets = targets
        self.transform = transform
        self.cutout = Cutout(n_holes=1, length=16)
        self.randomcrop = RandomCrop(32, padding=4)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform == 'flips':
            if np.random.binomial(1, 0.5) == 1:
                x = x.flip(2)
        elif self.transform == 'all':
            x = self.randomcrop(x)
            if np.random.binomial(1, 0.5) == 1:
                x = x.flip(2)
            x = self.cutout(x)

        y = self.targets[idx]
        return x.float(), y

    def __len__(self):
        return self.data.size(0)
