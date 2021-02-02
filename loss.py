import torch
from itertools import permutations


#signal distortion rate(sdr)
#est, src = [batch, leng_signal]
#THERE IS AN ERROR! DO NOT USE SDRLoss YET
def SDRLoss(est, src):
    dot = torch.sum(torch.mul(est, src), dim=1)
    #est_s = torch.sqrt(torch.sum(est**2, dim=1))
    #src_s = torch.sqrt(torch.sum(src**2, dim=1))
    est_s = torch.sum(est**2, dim=1)
    src_s = torch.sum(src**2, dim=1)
    sdr = -torch.div(dot, torch.mul(est_s, src_s)) #sdr = [batch]
    sdr[torch.isnan(sdr)] = 0
    return sdr


def PIT(loss, dic_est_src):
    """
    dic_est_src = {
        "est": [list of est tensors=[batch, time]], "src": [list of sources=[batch,time]]
    }
    loss means a loss function to use
    output is the min loss
    """
    est_set = dic_est_src["est"]
    src_set = dic_est_src["src"]
    batch = est_set[0].shape[0]
    num = len(est_set)

    est_permute = permutations(est_set)

    #fix src and permute est so that we can calculate all permutation of est and src
    #est_permute: list of permutation order(tuple), [(perm1), ..., (permk)]
    #perm i: tuple of permutation num i, ex) (0, 3, 1, 2)
    for i, permute in enumerate(est_permute):
        for j, (est, src) in enumerate(zip(permute, src_set)):
            if j==0:
                loss_temp = loss(est, src)
            else:
                loss_temp = torch.add(loss_temp, loss(est, src))

        if i==0:
            loss_permute = loss_temp
        else:
            #loss_permute = [batch]
            loss_permute = torch.minimum(loss_permute, loss_temp)

    return torch.sum(loss_permute)
