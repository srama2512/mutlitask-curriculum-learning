from tensorboardX import SummaryWriter
from torch.autograd import Variable
from argparse import Namespace
from utils import auc_score

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
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--gradient_clip', type=str2bool, default=True, help='Enable gradient clipping?')
parser.add_argument('--iters', type=int, default=200000, help='Number of training iterations')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument('--cuda', type=str2bool, default=True)
parser.add_argument('--logdir', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--strategy', type=int, default=0, \
        help='[0: Fixated easy, 1: Fixated hard, 2: Rigid joint, 3: 3D Generic baseline, 4: Cumulative curriculum, 5: On Demand Learning]')
parser.add_argument('--lr_schedule', type=int, default=60000, help='reduce learning rate by 10 after every N epochs')
parser.add_argument('--load_model', type=str, default='', help='continue training from this checkpoint')
parser.add_argument('--curriculum_update_every', type=int, default=1000, help='Curriculum update interval')

opts = parser.parse_args()

if len(opts.logdir) == 0:
    writer = SummaryWriter()
else:
    writer = SummaryWriter(log_dir=opts.logdir)

# Add misc to path
sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))

from DataLoader import DataLoader
from Models import ModelMatch
if opts.strategy == 0:
    from LearningStrategies import fixated_easy as getCurriculum
    print('Using Fixated Easy curriculum!')
elif opts.strategy == 1:
    from LearningStrategies import fixated_hard as getCurriculum
    print('Using Fixated Hard curriculum!')
elif opts.strategy == 2:
    from LearningStrategies import rigid_joint_learning as getCurriculum
    print('Using Rigid Joint curriculum!')
elif opts.strategy == 3:
    from LearningStrategies import generic_3d_baseline as getCurriculum
    print('Using 3D Generic Learning curriculum!')
elif opts.strategy == 4:
    from LearningStrategies import cumulative_curriculum as getCurriculum
    print('Using Cumulative learning curriculum!')
elif opts.strategy == 5:
    from LearningStrategies import on_demand_learning as getCurriculum
    print('Using On Demand Learning curriculum!')
else:
    raise NameError('Strategy #%s is not defined!'%(opts.strategy))

# Create the data loader
print('Creating the DataLoader')
loader = DataLoader(opts)

# Create the model
print('\nCreating the joint Model\n')
net = ModelMatch()
print(net)

if opts.load_model != '':
    net.load_state_dict(torch.load(opts.load_model))
    print('===> Loaded model from %s'%(opts.load_model))

dummy_input_left = Variable(torch.randn((1, 3, 101, 101)))
dummy_input_right = Variable(torch.randn((1, 3, 101, 101)))
dummy_output = net(dummy_input_left, dummy_input_right)
writer.add_graph(net, dummy_output)

# Define the loss functions
criterion_match = nn.BCEWithLogitsLoss()
criterion_match_valid = nn.BCEWithLogitsLoss(size_average=False)

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opts.cuda:
    net = net.cuda()
    criterion_match = criterion_match.cuda()
    criterion_match_valid = criterion_match_valid.cuda()

def create_optimizer(net, lr, mom):
    optimizer = optim.SGD(net.parameters(), lr, mom)
    return optimizer

optimizer = create_optimizer(net, opts.lr, opts.momentum) 

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
    true_labels = []
    predicted_probs = []
    if task == 'match':
        while(not isExhausted):
            imgs_left, imgs_right, labels, isExhausted = loader.batch_match_valid(isPositive=True)
            total_size += labels.size()[0]
            
            imgs_left = Variable(imgs_left)
            imgs_right = Variable(imgs_right)
            labels = Variable(labels)
            labels = labels.float()

            if opts.cuda:
                imgs_left = imgs_left.cuda()
                imgs_right = imgs_right.cuda()
                labels = labels.cuda()
            
            pred = net.forward_match(imgs_left, imgs_right)
            loss_curr = criterion_match_valid(pred, labels)
            mean_loss += loss_curr.data[0]
            # Level specific loss computed only over the positive samples
            # since there are no levels in negative samples
            level_specific_loss[curr_idx] += loss_curr.data[0]
            # If current level samples are exhausted
            if total_size % samples_per_level == 0:
                curr_idx += 1
            true_labels.append(labels.data.cpu().numpy()[:, 0])
            predicted_probs.append(pred.data.cpu().numpy()[:, 0])

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
            labels = labels.float()

            pred = net.forward_match(imgs_left, imgs_right)
            loss_curr = criterion_match_valid(pred, labels)
            mean_loss += loss_curr.data[0]
            true_labels.append(labels.data.cpu().numpy()[:, 0])
            predicted_probs.append(pred.data.cpu().numpy()[:, 0])

    for i in range(len(level_specific_loss)):
        level_specific_loss[i] /= samples_per_level

    true_labels = np.hstack(true_labels)
    predicted_probs = np.hstack(predicted_probs)
    mean_loss /= total_size
    auc_value = auc_score(predicted_probs, true_labels) 
    # Set the network back to train mode
    net.train()
    return mean_loss, level_specific_loss, auc_value

# Start training            
iter_no = 0
net.train()

curriculum_opts = Namespace()
curriculum_opts.iter_no = 0
curriculum_opts.val_loss_levels = [1.0 for i in range(loader.nLevels)]
curriculum_opts.batch_size = opts.batch_size
curriculum_opts.nLevels = loader.nLevels
curriculum_opts.iters = opts.iters
valid_loss_best = 10000
current_lr = opts.lr

for iter_no in range(opts.iters):
    
    # Changing the learning rate schedule by creating a new optimizer
    if (iter_no + 1) % opts.lr_schedule == 0:
        print('===> Reducing Learning rate by 10')
        current_lr = current_lr / 2
        optimizer = create_optimizer(net, current_lr, opts.momentum)

    optimizer.zero_grad()

    curriculum_opts.iter_no = iter_no

    if (iter_no+1) % opts.curriculum_update_every == 0 or iter_no == 0:
        pose_curriculum = getCurriculum(curriculum_opts)
        # Match curriculum is half of pose_curriculum
        match_curriculum = [samples//2 for samples in pose_curriculum]
        if sum(match_curriculum) < opts.batch_size//2:
            match_curriculum[0] += opts.batch_size//2 - sum(match_curriculum)

    match_left, match_right, match_labels = loader.batch_match(match_curriculum)
    match_labels = match_labels.float()
    
    match_left, match_right, match_labels = Variable(match_left), Variable(match_right), Variable(match_labels)
    
    if opts.cuda:
        match_left, match_right, match_labels = match_left.cuda(), match_right.cuda(), match_labels.cuda()

    preds1 = net.forward(match_left, match_right)

    loss_match = criterion_match(preds1, match_labels)
    
    loss_match.backward()
   
    if opts.gradient_clip:
        # NOTE: 0.25 was selected arbitrarily. The authors have not provided the
        # gradient clipping value.
        torch.nn.utils.clip_grad_norm(net.parameters(), 0.1)
    optimizer.step()

    if (iter_no+1) % 50 == 0:
        print('===> Iteration: %5d,          Loss:%8.4f'%(iter_no+1, loss_match.data[0]))
        writer.add_scalar('data/train_loss', loss_match.data[0], iter_no)
        for name, param in net.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), iter_no)
    
    if (iter_no+1) % 1000 == 0 or iter_no == 0:
        match_loss, match_loss_per_level, match_auc = evaluate(net, loader, 'match', opts)

        curriculum_opts.val_loss_levels = match_loss_per_level
        if match_loss <= valid_loss_best:
            valid_loss_best = match_loss
            torch.save(net.state_dict(), os.path.join(opts.save_dir, 'model_best.net'))

        torch.save(net.state_dict(), os.path.join(opts.save_dir, 'model_latest.net'))
        writer.add_scalar('data/val_loss', match_loss, iter_no)
        writer.add_scalars('data/val_loss_levels', {"level-%d"%(i):match_loss_per_level[i] for i in range(len(match_loss_per_level))}, iter_no)
        writer.add_scalar('data/val_auc', match_auc, iter_no)

        print('Validation Match loss: %.4f,     Validation AUC score: %.4f'%(match_loss, match_auc))
        print('    '.join(['Lvl %d: %.4f'%(i, v) for i, v in enumerate(match_loss_per_level)]))

        print('   '.join(['Lvl %d: %.4f'%(i, v) for i, v in enumerate(match_loss_per_level)]))
        print(match_loss_per_level)


writer.close()
