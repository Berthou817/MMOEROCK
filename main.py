import numpy as np
import sys
import time
# from newmodel_re import *
from hard_4 import *
import torch.optim as optim
import torch
from torch import nn

from torch.utils.data import TensorDataset
from loss import AutomaticWeightedLoss
from sklearn.metrics import r2_score

train_data = np.load("train_real.npy").reshape((-1,1,256,256))

k= np.load("k_real.npy").reshape((-1,1))
miu = np.load("m_real.npy").reshape((-1,1))
vp = np.load("vp_real.npy").reshape((-1,1))
vs = np.load("vs_real.npy").reshape((-1,1))

train_data = torch.tensor(train_data).type(torch.FloatTensor)
k = torch.tensor(k).type(torch.FloatTensor)
miu = torch.tensor(miu).type(torch.FloatTensor)
vp = torch.tensor(vp).type(torch.FloatTensor)
vs = torch.tensor(vs).type(torch.FloatTensor)


data = TensorDataset(train_data,k,miu,vp,vs)

data_loader = torch.utils.data.DataLoader(data,batch_size=16,shuffle=True)


device = torch.device('cuda:0')
net = MMOE().to(device)
awl = AutomaticWeightedLoss(4)

optimizer = optim.Adam([
                {'params': net.parameters()},
                {'params': awl.parameters(), 'weight_decay': 0}
            ],lr=0.0001,betas=(0.5,0.999))
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5,verbose=True,min_lr=0)

i = 0
epochs = 200
mse = nn.MSELoss()
Loss_list_k = []
Loss_list_phi = []
Loss_list_miu = []
Loss_list_vp= []
Loss_list_vs = []
Loss_list = []


for epoch in range(epochs):
    train_loss_k = 0
    train_loss_p = 0
    train_loss_m = 0
    train_loss_vp= 0
    train_loss_vs = 0
    train_loss = 0
    loss1 = 0

    i=0
    net.train()
    t1 = time.time()
    for data,k_train,miu_train,vp_train,vs_train in data_loader:
        i += 1
        data = data.cuda()
        miu_train = miu_train.cuda()
        vp_train = vp_train.cuda()
        vs_train = vs_train.cuda()
        k_train = k_train.cuda()
        optimizer.zero_grad()
        pred_k,pred_m,pred_vp,pred_vs = net(data)


        loss_k = mse(pred_k,k_train)
        loss_miu = mse(pred_m,miu_train)
        loss_vp = mse(pred_vp,vp_train)
        loss_vs = mse(pred_vs,vs_train)
        loss = awl(loss_k,loss_miu,loss_vp,loss_vs)

        loss.backward()
        optimizer.step()

        train_loss_m += float(loss_miu.item())
        train_loss_k += float(loss_k.item())
        train_loss_vp += float(loss_vs.item())
        train_loss_vs += float(loss_vp.item())
        train_loss += float(loss.item())
        loss1 = train_loss_k + train_loss_m+train_loss_vp+train_loss_vs

        sys.stdout.write(
            "[EPoch %d/%d] [Batch:%d/%d] [loss: %f] [K loss: %f]  [M loss: %f][VP loss: %f]  [VS loss: %f][KR2: %f] [MR2: %f] [VPR2: %f] [VSR2: %f]\n" % (
            epoch, epochs, len(data_loader), i, loss.item(), loss_k.item(),loss_miu.item(), loss_vp.item(),loss_vs.item(), r2_score(k_train.data.cpu().numpy(),pred_k.data.cpu().numpy()), r2_score(miu_train.data.cpu().numpy(),pred_m.data.cpu().numpy())
            , r2_score(vp_train.data.cpu().numpy(),pred_vp.data.cpu().numpy()), r2_score(vs_train.data.cpu().numpy(),pred_vs.data.cpu().numpy())))

    t2 = time.time()
    print(t2-t1)
    Loss_list.append(train_loss / len(data_loader))
    Loss_list_miu.append(train_loss_m / len(data_loader))
    Loss_list_k.append(train_loss_k / len(data_loader))
    Loss_list_vp.append(train_loss_vp / len(data_loader))
    Loss_list_vs.append(train_loss_vs / len(data_loader))

    scheduler.step(loss1)
    if epoch % 20== 0 and epoch != 0:
        torch.save(net.state_dict(), "gray/net_%d.pth" % epoch)
        np.savetxt("gray/Loss_{}.csv".format(epoch), np.array(Loss_list))

        np.savetxt("gray/Loss_m_{}.csv".format(epoch), np.array(Loss_list_miu))
        np.savetxt("gray/Loss_k_{}.csv".format(epoch), np.array(Loss_list_k))
        np.savetxt("gray/Loss_vp_{}.csv".format(epoch), np.array(Loss_list_vp))
        np.savetxt("gray/Loss_vs_{}.csv".format(epoch), np.array(Loss_list_vs))




torch.save(net.state_dict(), "gray/net.pth")

np.savetxt("gray/Loss.csv", np.array(Loss_list))

np.savetxt("gray/Loss_m.csv", np.array(Loss_list_miu))
np.savetxt("gray/Loss_k.csv", np.array(Loss_list_k))
np.savetxt("gray/Loss_vp.csv", np.array(Loss_list_vp))
np.savetxt("gray/Loss_vs.csv", np.array(Loss_list_vs))
