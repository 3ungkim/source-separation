import torch
import torch.nn
import torch.nn.functional as F


class DPTNet(nn.Module):
    def __init__(self):
        super(DPTNet, self).__init__()

        self.conv1d = nn.Conv1d(L, N=64, 
