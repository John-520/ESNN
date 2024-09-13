# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:08:07 2021

@author: 29792
"""
#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN1d(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(CNN1d, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.feature_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))
        self.__in_features = 256
    def forward(self, x):
        x = self.feature_layers(x)
        return x


    def output_num(self):
        return self.__in_features

