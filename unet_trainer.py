# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:22:36 2021

@author: mcgoug01
"""
import unet
import KiTs_Pytorch as kpt
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import time


path = 'C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits21\\kits21\\data'
kits = kpt.KiTS21_Data(path,n=1)
kitsloader = DataLoader(dataset=kits,batch_size=2,shuffle=True)

model = unet.UNet(depth=5,in_channels=4,out_labels=3)

costs=[]
opt = optim.Adam(model.parameters(),lr=0.0001)
loss= nn.CrossEntropyLoss()
start = time.time()
epochs = 8
model.train()
for epoch in range(1):
    print(epoch)
    for x,y in kitsloader:
    # for x,y in kits:
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],-1)
        print(x.shape,y.shape)
        pred = model(x.float())
        output = loss(pred, y)
        opt.zero_grad()
        output.backward()
        opt.step()
    costs.append(float(output))