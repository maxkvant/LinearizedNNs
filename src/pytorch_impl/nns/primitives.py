import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def Conv(in_filters, out_filters, groups=1):
    conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
    conv.weight.data *= np.sqrt(3)
    return conv


class Normalize(nn.Module):
    def forward(self, input):
        return nn.functional.normalize(input, p=2, dim=1, eps=1e-8)


class ReLU2(nn.Module):
    C = np.sqrt(2)

    def forward(self, input):
        return F.relu(input, inplace=True) * self.C


class ResidualConnection(nn.Module):
    def __init__(self, *layers):
        super(ResidualConnection, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return (input + self.layers(input)) / 2.
