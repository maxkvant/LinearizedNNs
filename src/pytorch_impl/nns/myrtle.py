from .primitives import *


class Myrtle5(nn.Module):
    def __init__(self, num_classes=10, input_filters=3, num_filters=256, groups=1):
        super(Myrtle5, self).__init__()
        filters = num_filters * groups

        def Activation():
            return ReLU2()

        self.layers = nn.Sequential(
            Conv(input_filters, filters), Activation(),
            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=8, stride=8),

            Normalize(),
            Flatten()
        )
        self.classifier = nn.Linear(num_filters, num_classes, bias=True)

    def readout(self, x):
        return self.layers(x)

    def forward(self, x):
        x = self.readout(x)
        return self.classifier(x)


class Myrtle7(nn.Module):
    def __init__(self, num_classes=10, input_filters=3, num_filters=256, groups=1):
        super(Myrtle7, self).__init__()
        filters = num_filters * groups

        def Activation():
            return ReLU2()

        self.layers = nn.Sequential(
            Conv(input_filters, filters), Activation(),
            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters, filters, groups), Activation(),
            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters, filters, groups), Activation(),
            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=8, stride=8),

            Normalize(),
            Flatten()
        )
        self.classifier = nn.Linear(num_filters, num_classes, bias=True)

    def readout(self, x):
        return self.layers(x)

    def forward(self, x):
        x = self.readout(x)
        return self.classifier(x)


class Myrtle10(nn.Module):
    def __init__(self, num_classes=10, input_filters=3, num_filters=256, groups=1):
        super(Myrtle10, self).__init__()
        filters = num_filters * groups

        def Activation():
            return ReLU2()

        self.layers = nn.Sequential(
            Conv(input_filters, filters), Activation(),

            Conv(filters, filters, groups), Activation(),
            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters, filters, groups), Activation(),
            Conv(filters, filters, groups), Activation(),
            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters, filters, groups), Activation(),
            Conv(filters, filters, groups), Activation(),
            Conv(filters, filters, groups), Activation(),
            nn.AvgPool2d(kernel_size=8, stride=8),

            Normalize(),
            Flatten()
        )
        self.classifier = nn.Linear(num_filters, num_classes, bias=True)

    def readout(self, x):
        return self.layers(x)

    def forward(self, x):
        x = self.readout(x)
        return self.classifier(x)