import torch
from torch import nn
import warnings

warnings.filterwarnings("ignore")

"""
date 2022.04.05
by wuhx cnn model
"""
class CNNModel(nn.Module):
    def __init__(self, num_classes: int):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(64),  # BN (5*5*16)
            nn.LeakyReLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(2,stride=2,padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

        )
        self.drop_out = nn.Dropout(p=0., inplace=False)
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=2048,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)


    """
        Forward propagation section
    """
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        # x = self.drop_out(x)
        x = self.conv2(x)
        # x = self.drop_out(x)
        x = self.conv3(x)
        # x = self.drop_out(x)
        x = self.conv4(x)
        x = self.drop_out(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x








