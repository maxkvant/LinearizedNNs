from .primitives import *


class Myrtle5(nn.Module):
    def __init__(self, num_classes=10, input_filters=3, num_filters=256, groups=1):
        super(Myrtle5, self).__init__()
        filters = num_filters

        def Activation():
            return ReLU2()

        self.layers = nn.Sequential(
            Conv(input_filters, filters * groups), Activation(),
            Conv(filters, filters * 2, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters * 2, filters * 4, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters * 4, filters * 8, groups), Activation(),
            nn.AvgPool2d(kernel_size=8, stride=8),

            Flatten(),
            Normalize(filters * 8)
        )
        self.classifier = nn.Linear(filters * 8 * groups, num_classes, bias=True)

    def readout(self, x):
        return self.layers(x)

    def forward(self, x):
        x = self.readout(x)
        return self.classifier(x)


class Myrtle7(nn.Module):
    def __init__(self, num_classes=10, input_filters=3, num_filters=256, groups=1):
        super(Myrtle7, self).__init__()
        filters = num_filters

        def Activation():
            return ReLU2()

        self.layers = nn.Sequential(
            Conv(input_filters, filters * groups), Activation(),
            Conv(filters, filters * 2, groups),    Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters * 2, filters * 4, groups), Activation(),
            Conv(filters * 4, filters * 8, groups), Activation(),

            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters *  8, filters * 16, groups), Activation(),
            Conv(filters * 16, filters * 32, groups), Activation(),

            nn.AvgPool2d(kernel_size=8, stride=8),

            Flatten(),
            Normalize(filters * 32)
        )
        self.classifier = nn.Linear(filters * 32 * groups, num_classes, bias=True)

    def readout(self, x):
        return self.layers(x)

    def forward(self, x):
        x = self.readout(x)
        return self.classifier(x)


class Myrtle10(nn.Module):
    def __init__(self, num_classes=10, input_filters=3, num_filters=1, groups=1):
        super(Myrtle10, self).__init__()
        filters = num_filters

        def Activation():
            return ReLU2()

        self.layers = nn.Sequential(
            Conv(input_filters, filters * groups),  Activation(),
            Conv(filters, filters * 2, groups),     Activation(),
            Conv(filters * 2, filters * 4, groups), Activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters * 4,  filters * 8, groups),  Activation(),
            Conv(filters * 8,  filters * 16, groups), Activation(),
            Conv(filters * 16, filters * 32, groups), Activation(),

            nn.AvgPool2d(kernel_size=2, stride=2),

            Conv(filters * 32, filters * 32, groups), Activation(),
            Conv(filters * 32, filters * 32, groups), Activation(),
            Conv(filters * 32, filters * 32, groups), Activation(),

            nn.AvgPool2d(kernel_size=8, stride=8),

            Flatten(),
            Normalize(filters * 32)
        )
        self.classifier = nn.Linear(filters * 32 * groups, num_classes, bias=True)

    def readout(self, x):
        return self.layers(x)

    def forward(self, x):
        x = self.readout(x)
        return self.classifier(x)
