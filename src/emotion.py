import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionNet(nn.Module):
    """Net which analyzes emotion"""
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()

        # 1st convolution layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)

        # 2nd convolution layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)

        # 3rd convolution layer
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        """Forward Process"""
        # 1st convolution layer
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # 2nd convolution layer
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        # 3rd convolution layer
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = x.view(-1, 128 * 2 * 2)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

# Instantiate the model
model = EmotionNet(num_classes=7)
