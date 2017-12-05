from tensorboardX import SummaryWriter
from torch.autograd import Variable
from argparse import Namespace

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import argparse
import torch
import sys
import pdb
import os

def ixvr(input_layer, bias_val=0.1):
    if not str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>":
        if hasattr(input_layer, 'weight'):
            nn.init.xavier_normal(input_layer.weight);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, bias_val);
    return input_layer

def inrml(input_layer, mean=0, std=0.01):
    if not str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>":
        if hasattr(input_layer, 'weight'):
            nn.init.normal(input_layer.weight, mean, std);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, 0.01);
    return input_layer

class LRN(nn.Module):
    """
    Implementation obtained from https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
    """
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1, padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
	return x

class MCNN(nn.Module):

    def __init__(self):
        super(MCNN, self).__init__()
        self.conv_shared = nn.Sequential(
                                ixvr(nn.Conv2d(3, 75, (7, 7), 1, 3)), # 75x224x224
                                #nn.BatchNorm2d(75),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d((3, 3)), # 75x74x74
                                LRN(local_size=5), 
                                ixvr(nn.Conv2d(75, 200, (5, 5), 1, 2)), # 200x74x74
                                #nn.BatchNorm2d(200),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d((3, 3)), # 200x24x24
                                LRN(local_size=5))

        """
        The conv3 groups are Gender, Nose, Mouth, Eyes, 
        Face and Others (AroundHead, FacialHair, Cheeks, Fat) 
        """
        for i in range(6):
            curr_conv3 = nn.Sequential(
                            ixvr(nn.Conv2d(200, 300, (3, 3), 1, 1)), # 300x24x24  
                            #nn.BatchNorm2d(300),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d((5, 5)), # 300x4x4
                            LRN(local_size=5))
            self.add_module('conv3%d'%(i), curr_conv3)

        self.fc_num_classes = [1, 2, 4, 5, 6, 13, 5, 2, 2]

        """
        The fc layers divided into 9 groups:
        Gender, Nose, Mouth, Eyes, Face, AroundHead, FacialHair, Cheeks, Fat

        (1) Gender    : Male
        (2) Nose      : Big Nose, Pointy Nose
        (4) Mouth     : Big Lips, Lipstick, Mouth Slightly Open, Smiling
        (5) Eyes      : Arched Eyebrows, Bags Under Eyes, Bushy Eyebrows, Eyeglasses, Narrow Eyes
        (6) Face      : Attractive, Blurry, Heavy Makeup, Oval Face, Pale Skin, Young
        (13) AroundHead: Balding, Bangs, Black Hair, Blond Hair, Brown Hair, Earrings, Gray Hair, Hat, Necklace, Necktie, Receding Hairline, Straight Hair, Wavy Hair
        (5) FacialHair: 5 o'clock Shadow, Goatee, Mustache, No Beard, Sideburns
        (2) Cheeks    : High Cheekbones, Rosy Cheeks
        (2) Fat       : Chubby, Double Chin
        """
        for i in range(9):
            curr_fc = nn.Sequential(
                            ixvr(nn.Linear(4800, 512)),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.5),
                            ixvr(nn.Linear(512, 512)),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.5),
                            ixvr(nn.Linear(512, self.fc_num_classes[i])))

            self.add_module('fc%d'%(i), curr_fc)        

    def forward(self, x):
        x = self.conv_shared(x)
        g1 = []
        for i in range(6):
            g1.append(getattr(self, 'conv3%d'%(i))(x).view(-1, 4800))
        
        g2 = []
        for i in range(9):
            if i < 5:
                g2.append(getattr(self, 'fc%d'%(i))(g1[i]))
            else:
                g2.append(getattr(self, 'fc%d'%(i))(g1[5]))

        return torch.cat(g2, dim=1)
