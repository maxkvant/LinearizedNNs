import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes, input_channels, num_channels=64, channel_groups=1):
        super(CNN, self).__init__()

        self.prepare = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),  # output: 32 x 32 x num_channels
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=channel_groups, bias=False),  # output: 32 x 32 x num_channels
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, groups=channel_groups, bias=False),  # output: 16 x 16 x num_channels
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(num_channels, num_classes, bias=False)

    def forward(self, x):
        x = self.prepare(x)
        x = self.layers(x)
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        return self.classifier(x_avg)
