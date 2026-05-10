import torch
import torch.optim as optim
import numpy as np
from Rmse_loss import rmse_loss
device = torch.device("cuda:0")
import time
from statistics import mean

def train_fun(net,x,y,N_red_lr=2,epochs=2000,threshold=0.0001,lr=0.001,printfun=True):
    if isinstance(x,np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y,np.ndarray):
        y = torch.from_numpy(y)
    old_time = time.time()
    net.train()
    check_loss = 1000
    Loss = []
    for j in range(N_red_lr):
        sss=0
        optimizer = optim.Adam(params=net.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            x = x.to(device)
            loss = rmse_loss(net(x),y)
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
            if check_loss>loss:
                check_loss = loss.item()
            if printfun:
                print(epoch,loss.item())
            if epoch>50 and (epoch+1)%50==0:
                try:
                    if printfun:
                        print(mean(Loss[-100:-50]),mean(Loss[-50:]),(mean(Loss[-100:-50])-mean(Loss[-50:]))/mean(Loss[-100:-50]))
                    if (mean(Loss[-100:-50])-mean(Loss[-50:]))/mean(Loss[-100:-50])<threshold:
                        sss = sss+1
                    if sss==1:
                        break
                except:
                    pass
        lr = lr/10
    new_time = time.time()
    print('运行时间：',new_time-old_time)
    print('loss:',Loss[-1])
    return check_loss,new_time-old_time,Loss