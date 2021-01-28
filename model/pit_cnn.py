import torch
import torch.nn as nn
import torch.nn.functional as F


class PITCNN(nn.Module):
    def __init__(self):
        super(PITCNN, self).__init__()

        self.layer1 = self._make_layer1(1, 64, 2)
        self.layer2 = self._make_layer2(4, 64)
        self.layer3 = self._make_layer1(64, 128, 2)
        self.layer4 = self._make_layer2(2, 128)
        self.layer5 = self._make_layer1(128, 256, 2)
        self.layer6 = self._make_layer2(2, 256)

    def _make_layer1(self, in_channel, out_channel, stride):
        layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        return layers

    def _make_layer2(self, num, channel_size):
        layers = []
        for i in range(num):
            layers.extend([
                nn.Conv2d(channel_size, channel_size, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(3, 2)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer5(x)
        out = self.layer6(x)
        return out
