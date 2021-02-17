import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, num_classes, num_features):
        super(FCN, self).__init__()
        self.num_features = num_features

        hidden_size = 256

        self.net = nn.Sequential(
          nn.Linear(num_features, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.num_features)
        return self.net(x)
