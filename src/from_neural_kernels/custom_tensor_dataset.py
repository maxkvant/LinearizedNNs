import torch
import numpy as np

from .transforms import Cutout, RandomCrop

"""
Based on part of the following code https://github.com/modestyachts/neural_kernels_code/blob/master/cifar10/run_cnn.py

"""


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    randomcrop = RandomCrop(32, padding=4)
    cutout     = Cutout(n_holes=1, length=16)

    def __init__(self, data, targets, transform=None):
        assert data.size(0) == targets.size(0)
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.augment_x(x, self.transform)

        y = self.targets[idx]
        return x, y

    def __len__(self):
        return self.data.size(0)

    @staticmethod
    def augment_x(x, transform='all'):

        if transform == 'flips':
            if np.random.binomial(1, 0.5) == 1:
                x = x.flip(2)
        elif transform == 'all':
            x = CustomTensorDataset.randomcrop(x)
            if np.random.binomial(1, 0.5) == 1:
                x = x.flip(2)
            x = CustomTensorDataset.cutout(x)
        return x.float()
