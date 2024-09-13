# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:34:56 2023

@author: luzy1
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import argparse
import numpy as np
import os

import Shuffle as Shuffle
import make_data
from models import models

import datasets
import sys
import numpy as np
from scipy.io import loadmat
import time
from ulties_functions import *
import matplotlib
import matplotlib.pyplot as plt

from RAdam import RAdam

import logging
import nni

from nni.utils import merge_parameter


import make_data
def load_data(X_train, Y_train, X_test,Y_test,X_test_t,Y_test_t
              ,batch_size,shuffle=True):
    loader_src = make_data.Make_data(X_train,Y_train,batch_size,shuffle=shuffle)
    loader_tar = make_data.Make_data(X_test,Y_test,batch_size,shuffle=shuffle)
    loader_tar_test = make_data.Make_data(X_test_t,Y_test_t,batch_size,shuffle=shuffle)
    return loader_src, loader_tar, loader_tar_test




for tasks in range(1):

    
        aa = []
        bb = []
        cc = []
        aa_processing = []
        aa_loss = []
        
        args = get_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            warnings.warn("gpu is not available")
            device = torch.device("cpu")
        
        SEED = args.seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
     
        
     
        
        
        '输入数据'
        # S_train_data 源域训练数据
        # T_train_data 目标域训练数据
        # T_test_data 目标域测试数据
         
         
        dataloaders = load_data(S_train_data, S_train_label, T_train_data,T_train_label,
                                T_test_data,T_test_label,
                            args.batch_size,shuffle=True)
            
    
    
        
        
        
        'KMC策略'#####################################################################
        from scipy.signal import freqz, stft, firwin
        import scipy
        '低通滤波器  模拟包络线'
        fs = 20000
        convld_kernel = 2
        H_env = firwin(convld_kernel, 6000, pass_zero = "lowpass", fs=fs)
        
        H_env_data = torch.from_numpy(np.expand_dims(np.expand_dims(H_env, axis=-1),axis=-1).swapaxes(0, 2)).float()    
        
        
        
        '多组带通滤波器  模拟FFT'
        num_filt = 600      
        num = 800                      
        convld_filters = num_filt
        convld_kernel = num
        
        bw = 0.5*fs/(convld_filters)       #生成的频率点最大为  0.5*fs
        H = np.ndarray(shape=(convld_filters,convld_kernel))
        for ii in range(convld_filters):
            H[ii] = firwin(convld_kernel, [bw*ii+0.01, bw*(ii+1)-0.01], pass_zero=False,fs=fs)
        H_data = torch.from_numpy(np.expand_dims(H, axis=-1).swapaxes(1, 2)).float()      
        
        
        model = models(args, H_env_data, H_data)
        model.to(device)
        
        
        
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            # 也可以判断是否为conv2d，使用相应的初始化方式 
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             # 是否为批归一化层
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                

        model.apply(weight_init)
        model.feature_layers.initialize_weights()



        correct = 0
        stop = 0
        
        
        optimizer = RAdam([
            {'params': model.feature_layers.layer_env.parameters(),'lr': 1e-5},     #1e-5
            {'params': model.feature_layers.layer.parameters(),'lr': 1e-5},
            {'params': model.feature_layers.feature_layers.parameters(),'lr': args.lr},
                {'params': model.bottle.parameters(), 'lr': args.lr},
            {'params': model.cls_fc.parameters(), 'lr': args.lr},
        ], lr=args.lr, weight_decay=args.decay) 
        
        
        
        time_start = time.time() #开始计时
        
        
        
        label_cha = 1
        labels = 1
        epoch_epoch = args.nepoch
        for epoch in range(1, args.nepoch + 1):
            stop += 1
            
            
          
            '训练和测试环节'
            train_loss_list, train_acc_sum = train_epoch(labels, label_cha,epoch_epoch, epoch, model, dataloaders, optimizer,i)
            print(f'Epoch: [{epoch:2d}]')
            t_correct = test(model, dataloaders[-1])
            
            '保存训练损失和测试精度结果'
            a_processing = 100. * t_correct / len(dataloaders[-1].dataset)
            aa_processing.append(a_processing)
            a_loss = sum(train_loss_list).cpu().data.numpy()
            aa_loss.append(a_loss)
            
            
            
            torch.save(model, 'pseudo_model.pkl')    
            if t_correct > correct:
                correct = t_correct
                stop = 0 
                torch.save(model, 'model.pkl')

            print(
                f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')
    
            if stop >= args.early_stop:
                print(
                    f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
                break
            
            
            
            
            
            
            
        time_end = time.time()    #结束计时
        time_c= time_end - time_start   #运行所花时间
        print('time cost', time_c, 's')
            
            
        a = 100. * correct / len(dataloaders[-1].dataset)
        
        model_save = torch.load('model.pkl')
        aa.append(a)


        with open('results.txt','a') as file0:
            print(snr_value, [i],np.mean(aa),time_c, file=file0)
            
            
    
    
        'loss和acc结果可视化'
        plt.figure()
        plt.plot(aa_processing)
        plt.show()
        plt.figure()
        plt.plot(aa_loss)
        plt.show()                
        
        
        '定时清楚cuda内存'
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        
        


            







                    