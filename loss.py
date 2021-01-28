import torch


#signal distortion rate(sdr)
#est, src = [batch, leng_signal]
def SDRLoss(est, src):
    dot = torch.sum(torch.mul(est, src), dim=1)
    est_s = torch.sqrt(torch.sum(src**2, dim=1))
    src_s = torch.sqrt(torch.sum(src**2, dim=1))
    sdr = -torch.div(dot, torch.mul(est_s, src_s)) #sdr = [batch]
    return sdr


def pit(loss, dic_est_src):
    """
    dic_est_src = {
        "est": [list of est tensors=[batch, time]], "src": [list of sources=[batch,time]]
    }
    loss means a loss function to use
    output is the min loss
    """
    est_set = dic_est_src["est"]
    src_set = dic_est_src["src"]

    #est_permute 필요
    #est_permute = 

    #fix src and permute est so that we can calculate all permutation of est and src
    #est_list = [list of est tensor=[batch, time]]
    for i, est_list in enumerate(est_permute):
        batch = est_list[0].shape[0]
        loss_temp = torch.zeros(batch)
        for est, src in zip(est_list, src_set):
            loss_temp = torch.add(loss_temp, loss(est, src))

        if i==0:
            loss_permute = loss_temp
        else:
            #loss_permute = [batch]
            loss_permute = torch.minimum(loss_permute, loss_temp)

    return torch.sum(loss_permute)
