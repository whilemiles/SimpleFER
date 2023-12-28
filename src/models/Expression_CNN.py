"""
inputs(48*48*1) ->
conv(24*24*64) -> conv(12*12*128) -> conv(6*6*256) ->
Dropout -> fc(4096) -> Dropout -> fc(1024) ->
outputs(7)
"""
from torch import nn


def gaussian_weights_init(m):
    """gaussian_weights_init"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


class ExpressionCNN(nn.Module):
    """ExpressionCNN"""
    def __init__(self):
        super(ExpressionCNN, self).__init__()

        # layer1(conv + relu + pool)
        # input:(batch_size, 1, 48, 48), output(batch_size, 64, 24, 24)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # layer2(conv + relu + pool)
        # input:(batch_size, 64, 24, 24), output(batch_size, 128, 12, 12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # layer3(conv + relu + pool)
        # input: (batch_size, 128, 12, 12), output: (batch_size, 256, 6, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256*6*6, 4096),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.RReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.RReLU(inplace=True),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        """forward"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y
