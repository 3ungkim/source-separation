import torch

def max_regul(src):
    abs_src = torch.abs(src)
    max_src = torch.max(abs_src)
    src = src / max_src
    return src
