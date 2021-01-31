import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channel=1, out_channel=24, size_conv=15):
        super(DownBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.size_conv = size_conv
        self.conv = nn.Conv1d(in_channel, out_channel, size_conv)

    def forward(self, x):
        #conv1d
        out1 = self.conv(x)
        #decimate
        out2 = out1[:,:,0::2]
        return out1, out2

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, size_conv=5):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.size_conv = size_conv
        self.conv = nn.Conv1d(in_channel, out_channel, size_conv)
    
    def forward(self, x, dsblock):
        #upsample
        #upsample된 크기의 tensor 만들고 tensor[:,:,0::2] = 기존 tensor[:,:,1:2] = interpolation result 해서 하나로 만들기
        #concat
        out = self.conv(x)

class WaveUNet(nn.Module):
    def __init__(self, layer_num, size_conv):
        super(WaveUNet, self).__init__()
        self.layer_num = layer_num
        self.size_conv = size_conv

    def forward(self, mix):
