# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:22:36 2021

@author: mcgoug01
"""
import unet
import KiTs_Pytorch as kpt

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


labels = 4
trained = False
path = '@@@@@@@@@'
kits = kpt.KiTS21_Data(path,n=1)
kitsloader = DataLoader(dataset=kits,batch_size=2,shuffle=True)
model_loc = os.path.join(os.getcwd(),'unet')
print("")
print("")

if os.path.exists(model_loc): 
    model = torch.load(model_loc)
    print("Model loaded!")
else: model = unet.UNet(depth=5,in_channels=4,out_labels=labels)

costs=[]
opt = optim.Adam(model.parameters(),lr=0.0001)
#this line taken from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
pytorch_total_params = sum(p.numel() for p in model.parameters())
#scaling factor for loss is number of model parameters*number of input data
scaler = pytorch_total_params*512*512
print("Total number of model parameters (in millions):",pytorch_total_params/1e6)
print("")
loss= nn.CrossEntropyLoss()
start = time.time()
epochs = 8
model.train()
for epoch in range(1):
    bar = tqdm(kitsloader)
    for idx,(x,y) in enumerate(bar):
        bar.set_description("Training from KiTS")
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],-1)
        pred = model(x.float())
        output = loss(pred, y)
        opt.zero_grad()
        output.backward()
        opt.step()

        costs.append(float(output/scaler))
    torch.save(model,model_loc)
        
plt.plot(costs)
