from torch.autograd import Variable
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
parser.add_argument('--h5_path', type=str, default='dataset/train/regTrain/prepro.h5', help='path to preprocessed h5 file')
parser.add_argument('--json_path', type=str, default='dataset/train/regTrain/prepro.json', help='path to preprocessed json file')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--gradient_clip', type=str2bool, default=True, help='Enable gradient clipping?')
parser.add_argument('--iters', type=int, default=200000, help='Number of training iterations')
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
criterion_match_valid = nn.BCEWithLogitsLoss(size_average=False)

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opts.cuda:
    net = net.cuda()
    #criterion_pose = criterion_pose.cuda() 
    criterion_match = criterion_match.cuda()
    criterion_match_valid = criterion_match_valid.cuda()

optimizer = optim.SGD(net.parameters(), lr=opts.lr, momentum=opts.momentum)

def evaluate(net, loader, task, opts):
    net.eval()
    isExhausted = False
    mean_loss = 0
    total_size = 0
    level_specific_loss = [0, 0, 0, 0, 0]
    # NOTE: Making use of the fact that the batches do not contain samples from more than
    # one level. This is true only if the batch_size is a factor of the number of 
    # valid samples per level. Default batch_size: 250, Default samples per level: 1000
    samples_per_level = loader.nPosValid // loader.nLevels

    curr_idx = 0
    if task == 'pose':
        while(not isExhausted):
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
            level_specific_loss[curr_idx] += loss_curr
            # If current level samples are exhausted
            if total_size % samples_per_level == 0:
                curr_idx += 1
    else:
        while(not isExhausted):
            imgs_left, imgs_right, labels, isExhausted = loader.batch_match_valid(isPositive=True)
            total_size += labels.size()[0]
            
            imgs_left = Variable(imgs_left)
            imgs_right = Variable(imgs_right)
            labels = Variable(labels)

            if opts.cuda:
                imgs_left = imgs_left.cuda()
                imgs_right = imgs_right.cuda()
                labels = labels.cuda()
            
            pred = net.forward_match(imgs_left, imgs_right)
            loss_curr = criterion_match_valid(pred, labels)
            mean_loss += loss_curr
            # Level specific loss computed only over the positive samples
            # since there are no levels in negative samples
            level_specific_loss[curr_idx] += loss_curr
            # If current level samples are exhausted
            if total_size % samples_per_level == 0:
                curr_idx += 1

        isExhausted = False
        while(not isExhausted):
            imgs_left, imgs_right, labels, isExhausted = loader.batch_match_valid(isPositive=False)
            total_size += labels.size()[0]
            if opts.cuda:
                imgs_left = imgs_left.cuda()
                imgs_right = imgs_right.cuda()
                labels = labels.cuda()
            imgs_left = Variable(imgs_left)
            imgs_right = Variable(imgs_right)
            labels = Variable(labels)

            pred = net.forward_match(imgs_left, imgs_right)
            loss_curr = criterion_match_valid(pred, labels)
            mean_loss += loss_curr

    for i in range(len(level_specific_loss)):
        level_specific_loss[i] /= samples_per_level

    mean_loss /= total_size
    # Set the network back to train mode
    net.train()
    return mean_loss, level_specific_loss
            
iter_no = 0
best_validation_loss = 100000000.0 # Simply assigning large value to signify Infinity
net.train()

for iter_no in range(opts.iters):
    
    optimizer.zero_grad()
    pose_curriculum = [opts.batch_size // loader.nLevels for i in range(loader.nLevels)]
    match_curriculum = [opts.batch_size // (2*loader.nLevels) for i in range(loader.nLevels)]
    pose_left, pose_right, pose_labels = loader.batch_pose(pose_curriculum)
    match_left, match_right, match_labels = loader.batch_match(match_curriculum)
    match_labels = match_labels.float()

    temp_list = [pose_left, pose_right, pose_labels, match_left, match_right, match_labels]
    temp_list = [Variable(element) for element in temp_list]
    
    if opts.cuda:
        temp_list = [element.cuda() for element in temp_list]
    
    inputs = {'pose': temp_list[0:2], 'match': temp_list[3:5]}
    preds = net.forward(inputs)

    loss_pose = criterion_pose(preds[0], temp_list[2]) 
    loss_match = criterion_match(preds[1], temp_list[5])

    # The total loss is loss_pose + opts.loss_lambda * loss_match
    # This means that you have to backpropagate 1 via loss_pose and
    # opts.loss_lambda via loss_match
    
    # Refer to https://discuss.pytorch.org/t/multiple-output-tutorial-examples/3050/5 
    # for this particular form of backpropagation
    torch.autograd.backward([loss_pose, loss_match], \
                            [loss_pose.data.new(1).fill_(1), \
                             loss_match.data.new(1).fill_(opts.loss_lambda)])

    if opts.gradient_clip:
        # NOTE: 0.25 was selected arbitrarily. The authors have not provided the
        # gradient clipping value.
        torch.nn.utils.clip_grad_norm(net.parameters(), 0.25)

    optimizer.step()

    if iter_no % 100 == 0:
        print('===> Iteration: %5d, Loss:%5.4f'%(iter_no+1, loss_pose.data[0] +\
                                                opts.loss_lambda * loss_match.data[0]))


