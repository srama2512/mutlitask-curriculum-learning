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

def str2bool(s):
    return s.lower() in ('yes', 'true', 't', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--h5_path', type=str, default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal/data.h5', help='preprocessed dataset path')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--
