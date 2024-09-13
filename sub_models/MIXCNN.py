# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:26:50 2023

@author: luzy1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class DepthwiseConv1D(nn.Module):
    def __init__(self, dim_in, kernel_size, dilation_rate, depth_multiplier, padding="same",
                                       use_bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim_in * depth_multiplier, kernel_size=kernel_size, stride=1, padding=padding, groups=dim_in,
                              bias=use_bias, dilation=dilation_rate)

    def forward(self, x):
        x = self.conv(x)
        return x

class Mixconv(nn.Module):
    def __init__(self, channal=64, kersize=64, m=1, c=1, dim_in=128):
        super(Mixconv, self).__init__()
        self.depth_conv_1 = DepthwiseConv1D(dim_in=dim_in, kernel_size=kersize, dilation_rate=m, depth_multiplier=c, padding="same",
                                       use_bias=False)
        self.act_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(dim_in * m)
        self.conv_1 = nn.Conv1d(dim_in * m, channal, kernel_size=1, stride=1, padding="same")
        self.act_3 = nn.ReLU()
        self.bn_3 = nn.BatchNorm1d(channal)

    def forward(self, x):
        x1 = x
        x = self.depth_conv_1(x)
        x = self.act_2(x)
        x = self.bn_2(x)
        x = torch.add(x, x1)
        x = self.conv_1(x)
        x = self.act_3(x)
        x = self.bn_3(x)
        return x




######################################################################################################################
class MIXCNN(nn.Module):
    def __init__(self,pretrained=False):
        super(MIXCNN, self).__init__()
        # self.conv_1 = nn.Conv1d(1, 128, kernel_size=32, stride=4)
        # self.bn_1 = nn.BatchNorm1d(128)
        # self.act_1 = nn.ReLU()
        # self.mix_1 = Mixconv(dim_in=128, channal=128, kersize=64, m=1, c=1)
        # self.mix_2 = Mixconv(dim_in=128, channal=128, kersize=64, m=1, c=1)
        # self.mix_3 = Mixconv(dim_in=128, channal=128, kersize=64, m=1, c=1)
        # self.bn_2 = nn.BatchNorm1d(128)
        # self.act_2 = nn.ReLU()
        # self.pool = nn.AdaptiveAvgPool1d(8)
        # self.fc = nn.Linear(128, 10)
        
        self.conv_1 = nn.Conv1d(1, 16, kernel_size=32, stride=4)
        self.bn_1 = nn.BatchNorm1d(16)
        self.act_1 = nn.ReLU()
        self.mix_1 = Mixconv(dim_in=16, channal=32, kersize=3, m=1, c=1)
        self.mix_2 = Mixconv(dim_in=32, channal=64, kersize=3, m=1, c=1)
        self.mix_3 = Mixconv(dim_in=64, channal=128, kersize=3, m=1, c=1)
        self.bn_2 = nn.BatchNorm1d(128)
        self.act_2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Linear(128, 10)
        
        
        self.__in_features = 1024
        
        
    def forward(self, x):
        x = self.conv_1(x)
        x = F.pad(x, (387, 388), "constant", 0)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.mix_1(x)
        x = self.mix_2(x)
        x = self.mix_3(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        x = self.pool(x).squeeze()
        # x = self.fc(x)
        return x



    # def forward(self, x):
    #     x = self.feature_layers(x)
    #     return x


    def output_num(self):
        return self.__in_features





if __name__ == '__main__':
    input = torch.rand(2, 1, 1024).cuda()
    model = MIXCNN().cuda()
    output = model(input)
    print(output.size())
