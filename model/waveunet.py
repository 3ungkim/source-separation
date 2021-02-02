import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DownBlock(nn.Module):
    def __init__(self, in_channel=1, out_channel=24, size_conv=15):
        super(DownBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.size_conv = size_conv
        self.conv = nn.Conv1d(in_channel, out_channel, size_conv)
        #self.norm = nn.BatchNorm1d(out_channel, momentum=0.01)
        self.norm = nn.BatchNorm1d(out_channel)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        #conv1d
        out1 = self.conv(x)
        out1 = self.norm(out1)
        out1 = self.lrelu(out1)
        #decimate
        out2 = out1[:,:,0::2]
        return out1, out2


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, up_channel, size_conv=5):
        super(UpBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up_channel = up_channel
        self.size_conv = size_conv

        self.upsample = nn.Conv1d(up_channel, up_channel, 2)
        #self.sig = nn.Sigmoid()
        self.conv = nn.Conv1d(in_channel, out_channel, size_conv)
        self.norm = nn.BatchNorm1d(out_channel, momentum=0.01)
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, x, dscrop):
        #Upsample1
        #learned linear interpolation can be interpreted as conv, wft + (1-w)ft+1
        #conv_weight = self.sig(self.upsample.weight)
        #conv_regul = torch.sum(conv_weight, dim=-1, keepdim=True)
        #param = torch.div(conv_weight, conv_regul)
        #param[torch.isnan(param)] = 0
        #self.upsample.weight = nn.Parameter(param, requires_grad=True)

        interpol = self.upsample(x)
        
        #Upsample2
        #concatenate origin tensor and upsampled tensor(interpolated tensor)
        shape = list(x.shape)
        shape[-1] = shape[-1]*2 - 1
        out = torch.ones(shape)
        out[:, :, 0::2] = x
        out[:, :, 1::2] = interpol

        #Crop and Concat
        crop_first = int((dscrop.shape[2])/2 - (out.shape[2])/2)
        crop_last = int((dscrop.shape[2])/2 + (out.shape[2])/2)
        crop = dscrop[:, :, crop_first:crop_last]

        out = torch.cat((out, crop), dim=1)

        #Conv1d
        out = self.conv(out)
        out = self.norm(out)
        out = self.lrelu(out)

        return out


class WaveUNet(nn.Module):
    def __init__(self, layer_num=12, size_dsconv=15, size_usconv=5, channel_size=24, source_num=2):
        super(WaveUNet, self).__init__()
        self.layer_num = layer_num
        self.size_dsconv = size_dsconv
        self.size_usconv = size_usconv
        self.channel_size = channel_size

        self.dsblock_list = self._repeat_dsblock(
            layer_num=layer_num,
            size_conv=size_dsconv,
            channel_size=channel_size
        )
        self.conv1 = nn.Conv1d(
            in_channels=channel_size*layer_num,
            out_channels=channel_size*(layer_num+1),
            kernel_size=size_dsconv
        )
        self.norm1 = nn.BatchNorm1d(channel_size*(layer_num+1), momentum=0.01)
        self.lrelu = nn.LeakyReLU()
        self.usblock_list = self._repeat_usblock(
            layer_num=layer_num,
            size_conv=size_usconv,
            channel_size=channel_size
        )
        self.conv2 = nn.Conv1d(
            in_channels=channel_size+1,
            out_channels=source_num-1,
            kernel_size=1
        )
        self.norm2 = nn.BatchNorm1d(source_num-1, momentum=0.01)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        out = copy.deepcopy(x)

        #downsample block
        dscrop_list = []
        for dsblock in self.dsblock_list:
            dscrop, out = dsblock(out)
            dscrop_list.append(dscrop)

        #middle block
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.lrelu(out)

        #upsample block
        dscrop_list.reverse()
        for usblock, dscrop in zip(self.usblock_list, dscrop_list):
            out = usblock(out, dscrop)

        #final block
        crop_first = int((x.shape[2])/2 - (out.shape[2])/2)
        crop_last = int((x.shape[2])/2 + (out.shape[2])/2)
        crop = x[:, :, crop_first:crop_last]

        out = torch.cat((out, crop), dim=1)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.tanh(out)

        return torch.squeeze(out, dim=1)

    def _repeat_dsblock(self, layer_num=12, size_conv=15, channel_size=24):
        dsblock_list = nn.ModuleList([DownBlock(1, channel_size, size_conv)])
        for i in range(1, layer_num):
            dsblock_list.append(DownBlock(
                in_channel=channel_size*i,
                out_channel=channel_size*(i+1),
                size_conv=size_conv
            ))
        return dsblock_list

    def _repeat_usblock(self, layer_num=12, size_conv=5, channel_size=24):
        #UpBlock(in_channel, out_channel, up_channel, size_conv)
        usblock_list = nn.ModuleList()
        for i in range(layer_num, 0, -1):
            usblock_list.append(UpBlock(
                in_channel=channel_size*(2*i+1),
                out_channel=channel_size*i,
                up_channel=channel_size*(i+1),
                size_conv=size_conv
            ))
        return usblock_list
