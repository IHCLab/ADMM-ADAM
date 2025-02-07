#!/usr/bin/env python
# coding: utf-8
import time
from model_inpainting0519 import generator
import torch
import numpy as np
from ipdataset import *
import scipy.io as sio

# Hyperparameters
batch_size_par = 1
device = 'cuda'
TARGET = 'test'
 
def lmat(fn):
    data=loadmat(fn)
    data_value=data[list(data.keys())[-1]]
    return data_value
        
def loadTxt(fn):
    fns = []
    with open(fn, 'r') as fp:
        data = fp.readlines()
        for item in data:
            fn = item.strip('\n')
            fns.append(fn)
    return fns

valfn = loadTxt('HSI_dataset/val_%s.txt' % TARGET)  
val_loader = torch.utils.data.DataLoader(dataset_h5(valfn, mode='Validation'), batch_size=batch_size_par)

netG = generator()
netG = torch.nn.DataParallel(netG).to(device)
state_dictG = torch.load('TGRS_ADMMADAM.pth') 
netG.load_state_dict(state_dictG)

if device=='cuda':
    print('using gpu')
    
with torch.no_grad():
    loadmask = loadTxt('HSI_dataset/val_mask.txt')
    mask = lmat(loadmask[0]).astype(np.float)
    mask_tensor = torch.from_numpy(mask).permute(2,0,1).type(torch.FloatTensor).to(device)
    
    for i, (vl_data, vl_fn) in enumerate(val_loader):
        netG.eval()
        vl_data= vl_data.to(device).permute(0,3,1,2).float()
        corrupted_data = vl_data*mask_tensor
        
        start_time = time.time()
        val_netG_result = netG(corrupted_data)
        time_dl = time.time()-start_time

        ## Recovery to image HSI
        val_rec_DL = val_netG_result.permute(0,2,3,1).cpu().numpy()

    print('Time: %.3f' % time_dl)
    sio.savemat("results/gan.mat", {'gan':val_rec_DL[0],'time_dl': time_dl})