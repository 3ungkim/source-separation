import torch
import numpy as np
from loss import SDRLoss
from loss import pit
from data.mkdata import VCTKDataset
from torch.utils.data import DataLoader
import argparse


parser = argparse.ArgumentParser(description="type hyperparams")
parser.add_argument("--time", type=int, help="source length")
parser.add_argument("--sr", type=int, help="sampling rate")
parser.add_argument("--batch", type=int, help="batch size")
args = parser.parse_args()

time = args.time
sr = args.sr 
batch_size = args.batch

np.random.seed(27)
torch.manual_seed(27)
train_source1 = VCTKDataset(path_csv="./data/source_path.csv", time=time, sr=sr)
train_source2 = VCTKDataset(path_csv="./data/source_path.csv", time=time, sr=sr)
train_loader1 = DataLoader(train_source1, batch_size=batch_size, shuffle=True)
train_loader2 = DataLoader(train_source2, batch_size=batch_size, shuffle=True)

#model = 

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

for epoch in range(60):
    model.train()
    running_loss = 0.0
    #src_list = []
    #for i, data in enumerate(train_loader):
    #   if i%2==0:
    #       src_list.append(data)
    #       continue
    #   else:
    #       src_list.append(data)
    #mix = src_list[0] + src_list[1] 
    train1 = iter(train_loader1)
    train2 = iter(train_loader2)
    for i in range(2500):
        src1 = train1.next()
        src2 = train2.next()
        src_list = [src1, src2]
        mix = src1 + src2

        optimizer.zero_grad()

        est1, est2 = model(mix)
        est_list = [est1, est2]
        dic_est_src = {"est": est_list, "src": src_list}

        loss = pit(SDRLoss, dic_est_src)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%500==499:
            print(f'Train Epoch: {epoch+1}/60 {i+1}/2500\tLoss: {running_loss/500}')
            running_loss = 0.0

    scheduler.step()
