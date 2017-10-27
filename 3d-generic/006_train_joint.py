from torch.autograd import Variable
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
parser.add_argument('--h5_path', type=str, default='dataset/train/regTrain/prepro.h5', help='path to preprocessed h5 file')
parser.add_argument('--json_path', type=str, default='dataset/train/regTrain/prepro.json', help='path to preprocessed json file')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--gradient_clip', type=str2bool, default=True, help='Enable gradient clipping?')
parser.add_argument('--iter', type=int, default=200000, help='Number of training iterations')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument('--cuda', type=str2bool, default=True)
parser.add_argument('--loss_lambda', type=int, default=1)

opts = parser.parse_args()

# Add misc to path
sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))

from DataLoader import DataLoader
from Models import ModelJoint
# Create the data loader
print('Creating the DataLoader')
loader = DataLoader(opts)

# Create the model
print('\nCreating the joint Model\n')
net = ModelJoint()
print(net)

'''
# Define the loss functions
def criterion_pose(pred, labels, size_average=True):
    # Compute L2 norm squared
    e2 = (pred - labels)*(pred - labels)
    e2 = e2.sum(1)
    if size_average:
        return torch.mean(torch.log(F.relu(e2-1)+1) - F.relu(1-e2) + 1)
    else:
        return torch.sum(torch.log(F.relu(e2-1)+1 - F.relu(1-e2) + 1))

criterion_match = nn.BCEWithLogitsLoss()

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opts.cuda:
    net = net.cuda()
    #criterion_pose = criterion_pose.cuda() 
    criterion_match = criterion_match.cuda()

optimizer = optim.SGD(net.parameters(), lr=opts.lr, momentum=opts.momentum)

def evaluate(net, loader, task, opts):
    net = net.evaluate()
    isExhausted = False
    mean_loss = 0
    total_size = 0
    if task == 'pose':
        while(!isExhausted):
            imgs_left, imgs_right, labels, isExhausted = loader.batch_pose_valid()
            total_size += labels.size()[0]
            if opts.cuda:
                imgs_left = imgs_left.cuda()
                imgs_right = imgs_right.cuda()
                labels = labels.cuda()
            imgs_left = Variable(imgs_left)
            imgs_right = Variable(imgs_right)
            labels = Variable(labels)

            pred = net.forward_pose(imgs_left, imgs_right) 
            loss_curr = criterion_pose(pred, labels, size_average=False)
            mean_loss += loss_curr
    else

            

'''
