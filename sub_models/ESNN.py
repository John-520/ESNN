# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:16:48 2023

@author: luzy1
"""
from torch import nn
import warnings
import torch
import torch.nn.functional as F
        
        


class ESNN(nn.Module):
    def __init__(self, args, H_env_data, H_data, pretrained=False,  in_channel=1, out_channel=5):
        super(ESNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")
        self.args = args
        self.layer_env = nn.Conv1d(1, H_env_data.shape[0] ,H_env_data.shape[2], stride=1, padding='same')
        self.layer = nn.Conv1d(1, H_data.shape[0], kernel_size=H_data.shape[2], stride=1,padding=1)
        self.AMP = nn.AdaptiveMaxPool1d(1)
               
        
        self.weight_env = H_env_data.cuda()                                   
        self.bias_env = torch.zeros([ self.weight_env.shape[0] ]).cuda()      
        
        self.weight = H_data.cuda()
        self.bias = torch.zeros([ self.weight.shape[0] ]).cuda()
        
        
        
        
        self.feature_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=60, stride=6,padding=1,dilation=1),  # 16, 26 ,26  
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2,padding=1),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1,padding=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )
        
        self.__in_features = 320 #

    
        
    def forward(self, x):
        
        x = envelope_data(x)
        x = self.layer_env(x)
        x = self.layer(x)
        x = self.AMP(x)

        x = x.swapaxes(1, 2)
        x = self.feature_layers(x)

        return x




    def output_num(self):
        return self.__in_features



    def initialize_weights(self):
        
        self.layer.weight = nn.Parameter(self.weight)
        self.layer.bias = nn.Parameter(self.bias)
        self.layer.requires_grad = False
        
                
        self.layer_env.weight = nn.Parameter(self.weight_env)
        self.layer_env.bias = nn.Parameter(self.bias_env)
        self.layer_env.requires_grad = False
        
        
        for param in self.layer.parameters():
            print(param, param.requires_grad)
            param.requires_grad = True
            print(param.requires_grad)
        
        for param in self.layer_env.parameters():
            print(param, param.requires_grad)
            param.requires_grad = True
            print(param.requires_grad)
        
        
        
                





def envelope_data(en):
    en_raw_abs = en - torch.mean(en)
    es_raw_abs = torch.abs(en_raw_abs) #* 2 / len(en_raw_abs)
    return es_raw_abs












