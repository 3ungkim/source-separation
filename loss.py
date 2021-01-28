import torch


#signal distortion rate(sdr)
#est, target = [batch, leng_signal]
def SDRLoss(est, target):
    dot = torch.sum(torch.mul(est, target), dim=1)
    est_s = torch.sqrt(torch.sum(target**2, dim=1))
    target_s = torch.sqrt(torch.sum(target**2, dim=1))
    sdr = -torch.div(dot, torch.mul(est_s, target_s)) #sdr = [batch]
    sdr_sum = torch.sum(sdr)
    return sdr_sum

def pit(loss):
