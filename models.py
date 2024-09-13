# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:12:18 2021

@author: 29792
"""
import numpy as np
import torch
import torch.nn as nn
from loss.DAN import DAN
from loss.MMD import mmd_rbf_noaccelerate, mmd_rbf_accelerate
from loss.JAN import JAN
from loss.MMD_loss import MMD_loss
from loss.CORAL import CORAL
import sys 
sys.path.append("D:\北京交通大学博士\论文【小】\论文【第四章】\code") 
from MMSD_main.MMSD import MMSD

from loss.lmmd import LMMD_loss
from loss.contrastive_center_loss import ContrastiveCenterLoss
from loss.SupervisedContrastiveLoss import SupervisedContrastiveLoss
from loss.ContrastiveLoss import ContrastiveLoss
import sub_models
import torch.nn.functional as F

from loss.adv import *


class models(nn.Module):

    def __init__(self, args, H_env_data, H_data):
        super(models, self).__init__()
        self.args = args
        self.H_data = H_data
        self.num_classes= args.num_classes
        #这'句话很重要'
        self.feature_layers = getattr(sub_models, args.model_name)(args, H_env_data, H_data, args.pretrained)
        self.bottle = nn.Linear(self.feature_layers.output_num(), 256) #
        self.cls_fc = nn.Linear(256, self.num_classes) #

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
        



    def forward(self, source,target,label_source,label_target,epoch,mu_value,task): #(64,1,1200)
        loss = 0
        adv_loss=0
        dist_loss_mmd , dist_loss_jmmd= 0, 0
        
        
        source = self.feature_layers(source)           #(64,1,1200)---(64,1024,1)
        f_source = source.view(source.size(0), -1)
        source = self.bottle(f_source)
        s_pred = self.cls_fc(source)
            
        
        target = self.feature_layers(target)
        f_target = target.view(target.size(0), -1)
        target = self.bottle(f_target)
        t_pred = self.cls_fc(target)
        
        
        
        if self.training == True and epoch> self.args.middle_epoch:               

            'MMD'
            distance_loss = mmd_rbf_accelerate  #  
            dist_loss_mmd += distance_loss(source, target)

                
            'JMMD'
            self.softmax_layer = nn.Softmax(dim=1)
            self.softmax_layer = self.softmax_layer.to(self.device)
            distance_loss = JAN
            s_softmax_out = self.softmax_layer(s_pred)
            t_softmax_out = self.softmax_layer(t_pred)
            dist_loss_jmmd += distance_loss([source,s_softmax_out], [target, t_softmax_out])
                
            import dynamic_factor
            mu = dynamic_factor.estimate_mu(source.detach().cpu().numpy(), 
                                            torch.max(s_pred, 1)[1].detach().cpu().numpy(),
                                            target.detach().cpu().numpy(), 
                                            torch.max(t_pred, 1)[1].detach().cpu().numpy())
            
            
            
            loss =  (1-mu) * dist_loss_mmd +  mu * dist_loss_jmmd  
            
            
            if mu_value == 2:
                with open('results_miu.txt','a') as file0:
                    print([task],mu_value,np.mean(1-mu),np.mean(mu),file=file0)
            
            
        return s_pred, t_pred, loss, adv_loss




    def predict(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottle(x)
        return self.cls_fc(x)
    

    


