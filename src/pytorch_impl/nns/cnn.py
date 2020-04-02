import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes, input_channels, num_channels=64):
        super(CNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),             # output: 32 x 32
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),  # output: 32 x 32
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, bias=False),  # output: 16 x 16
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, bias=False),  # output:  8 x  8
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.classifier = nn.Linear(num_channels * 2, num_classes, bias=False)

    def forward(self, x):
        x = self.layers(x)
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)

        x = torch.cat([x_avg, x_max], dim=-1)

        return self.classifier(x)

