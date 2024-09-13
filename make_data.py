# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:37:11 2021

@author: 29792
"""
from functools import partial
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
import torch
from torchsampler import ImbalancedDatasetSampler



#################################解决num_works != 0时的堵塞问题   没啥用
import cv2
cv2.setNumThreads(0)
#################################



class Mydataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)
    def get_labels(self): 
        return self.y

def Make_data(X_train, Y_train,batch_size,shuffle=True):
    datasets = Mydataset(X_train, Y_train)  # 初始化

    dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=0) 
    return dataloader



# =============================================================================
# 
# class Mydataset(data.Dataset):
# 
#     def __init__(self, x):
#         self.x = x
#         self.idx = list()
#         for item in x:
#             self.idx.append(item)
#         pass
# 
#     def __getitem__(self, index):
#         input_data = self.idx[index]
#         return input_data
# 
#     def __len__(self):
#         return len(self.idx)
# 
# def Make_data(input_data,batch_size):
#     datasets = Mydataset(input_data)  # 初始化
# 
#     dataloader = data.DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=0) 
#     return dataloader
#     
# =============================================================================
    
# =============================================================================
# if __name__ ==('__main__'):
# 
#     datasets = Mydataset(X_train, Y_train)  # 初始化
# 
#     dataloader = data.DataLoader(datasets, batch_size=100, shuffle=True, num_workers=0) 
# 
#     for i, (input_data, target) in enumerate(dataloader):
# # =============================================================================
# #         print('input_data%d' % i, input_data)
# #         print('target%d' % i, target)
# # =============================================================================
#         print(input_data.shape,target.shape)
# =============================================================================



