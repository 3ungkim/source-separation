import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from loss import SDRLoss
from loss import PIT
from data.mkdata import VCTKDataset
from model.pit_cnn import PITCNN
from model.waveunet import WaveUNet
from utils import max_regul


parser = argparse.ArgumentParser(description="type hyperparams")
parser.add_argument("--time", type=int, help="source length")
parser.add_argument("--sr", type=int, help="sampling rate")
parser.add_argument("--batch", type=int, help="batch size")
parser.add_argument("--model", type=str, help="model type")
parser.add_argument("--lr", type=float, help="learning rate")
args = parser.parse_args()

time = args.time
sr = args.sr 
batch_size = args.batch
model_name = args.model
lr = args.lr


def train(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    train_source1 = VCTKDataset(path_csv="./data/source_path.csv", time=time, sr=sr)
    train_source2 = VCTKDataset(path_csv="./data/source_path.csv", time=time, sr=sr)
    train_loader1 = DataLoader(train_source1, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_source2, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    model.to(device)

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
            src1 = max_regul(src1)

            src2 = train2.next()
            src2 = max_regul(src2)

            mix = torch.add(src1, src2)

            optimizer.zero_grad()

            est1 = model(mix)

            crop_first = int((mix.shape[-1])/2 - (est1.shape[-1])/2)
            crop_last = crop_first + est1.shape[-1]

            mix = mix[:, crop_first:crop_last]
            est2 = mix - est1

            src1 = src1[:, crop_first:crop_last]
            src2 = src2[:, crop_first:crop_last]

            est_list = [est1, est2]
            src_list = [src1, src2]
            dic_est_src = {"est": est_list, "src": src_list}

            loss = PIT(SDRLoss, dic_est_src)
            print("loss", loss.item())
            
            loss.backward()
            optimizer.step()

            #for name, param in model.named_parameters():
                #if param.grad is not None:
                    #print("param grad", name, param.grad.sum())
                #else:
                    #print(name, param.grad)

            running_loss += loss.item()

            if i%100==99:
                print(f'Train Epoch: {epoch+1}/60 {i+1}/2500\tLoss: {running_loss/100}')

                if epoch==0 and i==99: #best_loss = running_loss
                    best_loss = running_loss
                elif best_loss > running_loss:
                    best_loss = running_loss
                    torch.save(model.state_dict(), "./model/saved_model.pt")
                else:
                    pass

                running_loss = 0.0

        scheduler.step()

if __name__=="__main__":
    np.random.seed(820)
    torch.manual_seed(820)

    if model_name=="waveunet":
        model = WaveUNet(layer_num=10, size_dsconv=15, size_usconv=5, channel_size=24, source_num=2)
        train(model)
    else:
        print("wrong model name")
