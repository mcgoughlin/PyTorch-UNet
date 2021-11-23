# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:22:36 2021

@author: mcgoug01
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np

#create and instantiate data-loading class
class KiTS21_Data(Dataset):
    def __init__(self, path, n:int=10, 
                 im:str="imaging.nii.gz", lab:str="aggregated_MAJ_seg.nii.gz",
                 transform=None):
        assert os.path.exists(path)
        
        samples = os.listdir(path)
        n = min([len(samples),n])
        
        self.data =[]
        for i in range(n):
            print(i)
            impath = os.path.join(os.path.join(path,samples[i]),im)
            segpath = os.path.join(os.path.join(path,samples[i]),lab)
            seg = nib.load(segpath).get_fdata()
            kidney_indices = np.where(np.argmax(seg.reshape(seg.shape[0],-1),1)>0)[0]
            for slice_ in kidney_indices:
                datum = {}
                datum['index'] = slice_
                datum['impath']=impath
                datum['segpath']=segpath
                self.data.append(datum)
            
        self.data=pd.DataFrame(self.data)
        self.len = self.data.shape[0]
        self.transform = transform
        
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx:int):
        #ensures consistent indexing using .iloc
        sample = self.data.iloc[idx]
        idx=sample['index']
        image = nib.load(sample['impath']).get_fdata()[idx]
        label = nib.load(sample['segpath']).get_fdata()[idx]
        label = F.one_hot(torch.tensor(label).long(),num_classes=3).float()
        if self.transform:
            return self.transform(image),label
        return image,label
    
def train(x,y,model,optimizer,loss_function,epochs=10):
    cost = []
    for epoch in range(epochs):
        total_loss =0
        for X,Y in zip(x,y):
            pred = model(X.float())
            loss = loss_function(pred,Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            #cumulative loss
            total_loss +=loss.item()
        cost.append(total_loss)
        
    return cost

if __name__ == "__main__":
    path = 'C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits21\\kits21\\data'
    dataset = KiTS21_Data(path,n=1)
    print(dataset[0])
    trainloader = DataLoader(dataset=dataset,batch_size=None,shuffle=True)