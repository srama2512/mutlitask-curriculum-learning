from utils import average_angular_error, average_translation_error
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
from Models import ModelPose
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

# Tensorboard vsualization output

# Create the data loader
print('Creating the DataLoader')
loader = DataLoader(opts)

# Create the model
print('\nCreating the Pose Model\n')
net = ModelPose()
print(net)
if opts.load_model != '':
    net.load_state_dict(torch.load(opts.load_model))
    print('===> Loaded model from %s'%(opts.load_model))

# Define the loss functions
def criterion_pose(pred, labels, size_average=True):
    # Compute L2 norm squared
    e2 = (pred - labels)*(pred - labels)
    e2 = e2.sum(1)
    if size_average:
        return torch.mean(torch.log(F.relu(e2-1)+1) - F.relu(1-e2) + 1)
        #return torch.mean(e2)
    else:
        return torch.sum(torch.log(F.relu(e2-1)+1 - F.relu(1-e2) + 1))
        #return torch.sum(e2)

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opts.cuda:
    net = net.cuda()

dummy_input_left = Variable(torch.randn((1, 3, 101, 101))).cuda()
dummy_input_right = Variable(torch.randn((1, 3, 101, 101))).cuda()
dummy_output = net(dummy_input_left, dummy_input_right)
writer.add_graph(net, dummy_output)

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
    if task == 'pose':
        saved_predictions = []
        saved_truth = []
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
            mean_loss += loss_curr.data[0]
            level_specific_loss[curr_idx] += loss_curr.data[0]
            # If current level samples are exhausted
            if total_size % samples_per_level == 0:
                curr_idx += 1

            pred_np = np.zeros((pred.size()[0], 6))
            labels_np = np.zeros((pred.size()[0], 6))
            pred_tmp = pred.data.cpu().numpy()
            labels_tmp = labels.data.cpu().numpy()
            pred_np[:, 0:2] = pred_tmp[:, 0:2]
            labels_np[:, 0:2] = labels_tmp[:, 0:2]
            pred_np[:, 3:] = pred_tmp[:, 2:]
            labels_np[:, 3:] = labels_tmp[:, 2:]

            saved_predictions.append(pred_np)
            saved_truth.append(labels_np)

        saved_predictions = np.vstack(saved_predictions)
        saved_truth = np.vstack(saved_truth)
        #NOTE: Assumes that only heading and pitch are being predicted. No roll. 
        aae = average_angular_error(saved_predictions[:, 0:3], saved_truth[:, 0:3])
        ate = average_translation_error(saved_predictions[:, 3:], saved_truth[:, 3:])
    
    for i in range(len(level_specific_loss)):
        level_specific_loss[i] /= samples_per_level

    mean_loss /= total_size
    # Set the network back to train mode
    net.train()
    return mean_loss, level_specific_loss, aae, ate
   
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
        print('===> Reducing Learning rate by 2')
        current_lr = current_lr / 2
        optimizer = create_optimizer(net, current_lr, opts.momentum)

    optimizer.zero_grad()
   
    curriculum_opts.iter_no = iter_no

    if (iter_no+1) % opts.curriculum_update_every == 0 or iter_no == 0:
        pose_curriculum = getCurriculum(curriculum_opts) 
    
    pose_left, pose_right, pose_labels = loader.batch_pose(pose_curriculum)
    pose_left, pose_right, pose_labels = Variable(pose_left), Variable(pose_right), Variable(pose_labels)
    
    if opts.cuda:
        pose_left, pose_right, pose_labels = pose_left.cuda(), pose_right.cuda(), pose_labels.cuda()

    preds0 = net.forward(pose_left, pose_right)
    loss_pose = criterion_pose(preds0, pose_labels) 
    loss_pose.backward()
   
    if opts.gradient_clip:
        # NOTE: 0.1 was selected arbitrarily. The authors have not provided the
        # gradient clipping value.
        torch.nn.utils.clip_grad_norm(net.parameters(), 0.1)
    optimizer.step()

    if (iter_no+1) % 50 == 0:
        print('===> Iteration: %5d,          Loss:%8.4f'%(iter_no+1, loss_pose.data[0])) 
        writer.add_scalar('data/train_loss', loss_pose.data[0], iter_no)
        for name, param in net.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), iter_no)
            
    if (iter_no+1) % 1000 == 0 or iter_no == 0:
        pose_loss, pose_loss_per_level, valid_aae, valid_ate = evaluate(net, loader, 'pose', opts) 
        
        curriculum_opts.val_loss_levels  = pose_loss_per_level
        if pose_loss <= valid_loss_best:
            valid_loss_best = pose_loss
            torch.save(net.state_dict(), os.path.join(opts.save_dir, 'model_best.net'))
       
        torch.save(net.state_dict(), os.path.join(opts.save_dir, 'model_latest.net'))
        writer.add_scalar('data/val_loss', pose_loss, iter_no)
        writer.add_scalars('data/val_loss_levels', {"level-%d"%(i): pose_loss_per_level[i] for i in range(len(pose_loss_per_level))}, iter_no)
        writer.add_scalar('data/average_angular_error', valid_aae, iter_no)
        writer.add_scalar('data/average_translation_error', valid_ate, iter_no)

        print('Pose loss: %.4f, AAE: %.4f, ATE: %.4f'%(pose_loss, valid_aae, valid_ate))
        print('   '.join(['Lvl %d: %.4f'%(i, v) for i, v in enumerate(pose_loss_per_level)]))
writer.close()
