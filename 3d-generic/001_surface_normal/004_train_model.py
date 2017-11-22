from tensorboardX import SummaryWriter
from torch.autograd import Variable
from argparse import Namespace
from utils import *

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import argparse
import torch
import json
import sys
import pdb
import os

def str2bool(s):
    return s.lower() in ('yes', 'true', 't', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal/data.h5', help='preprocessed dataset path')
parser.add_argument('--labels_path', type=str, default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal/normals.h5', help='preprocessed labels path')
parser.add_argument('--clusters_path', type=str, default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/001_surface_normal/temp_clusters.json')
parser.add_argument('--delaunay_path', type=str, default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/001_surface_normal/delaunay_vertices.json')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--cuda', type=str2bool, default=True)
parser.add_argument('--logdir', type=str, default='', help='path to save logs')
parser.add_argument('--save_dir', type=str, default='', help='path to save models')
parser.add_argument('--resume', type=str, default='', help='continue training from this checkpoint')
parser.add_argument('--pretrained', type=str, default='', help='path to pre-trained weights')
parser.add_argument('--random_seed', type=int, default=123)

opts = parser.parse_args()
print(opts)

torch.manual_seed(opts.random_seed)

def create_optimizer(net, opts):
    return optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    
if len(opts.logdir) == 0:
    writer = SummaryWriter()
else:
    writer = SummaryWriter(log_dir=opts.logdir)

# Add misc to path
sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))
from Models import ModelSurfaceNormal
from DataLoaderSN import DataLoader

# Create DataLoader
loader = DataLoader(opts)

# Create model
net = ModelSurfaceNormal()

# If pretrained weights available, load them
if opts.pretrained != '':
    pretrained_state = torch.load(opts.pretrained)
    net.load_weights(pretrained_state)

# If resume is enabled, start from previous model
if opts.resume != '':
    chkpt = torch.load(opts.resume)
    net.load_state_dict(chkpt)

criterion = nn.NLLLoss2d(reduce=False)

if opts.cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')
if opts.cuda:
    net = net.cuda()

#criterion = masked_cross_entropy_2d

if opts.pretrained != "":
    optimizer = create_optimizer(net.classifier, opts)
else:
    optimizer = create_optimizer(net, opts)
# This input size is designed to give 10x20x20x20 output
dummy_input = Variable(torch.randn((10, 3, 708, 738)))
if opts.cuda:
    dummy_input = dummy_input.cuda()

dummy_output = net(dummy_input)

writer.add_graph(net, dummy_output)

def evaluate(net, loader, split):

    net.eval()
    isExhausted = False
    predicted_classes_all = []
    true_classes_all = []
    masks_all = []
    while not isExhausted:
        images_curr, normals_curr, masks_curr, isExhausted = loader.batch_test(split)
        if opts.cuda:
            images_curr = images_curr.cuda()
            normals_curr = normals_curr.cuda()
            masks_Curr = masks_curr.cuda()

        images_curr = Variable(images_curr, requires_grad=True)
        normals_curr = Variable(normals_curr, requires_grad=True)
        masks_curr = Variable(masks_curr, requires_grad=True)
        preds_curr = net(images_curr)
        
        true_classes = normals_curr.cpu().data.numpy()
        predicted_classes = np.argmax(preds_curr.cpu().data.numpy(), axis=1)
        
        true_classes_all.append(true_classes)
        predicted_classes_all.append(predicted_classes)
        masks_all.append(masks_curr.cpu().data.numpy())

    predicted_classes_all = np.concatenate(predicted_classes_all, axis=0)
    true_classes_all = np.concatenate(true_classes_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)

    unbinned_score, binned_score = get_report(predicted_classes_all, true_classes_all, masks_all, 20) 
    net.train()
    
    return unbinned_score, binned_score

# Set the network to train mode
net.train()

epoch = 0
best_binned_score = 0
total_iters = loader.images['train'].shape[0] // opts.batch_size + 1
json.dump(vars(opts), open(os.path.join(opts.save_dir, 'opts.json'), 'w'))

for epoch in range(opts.epochs):

    isExhausted = False
    iter_no = 0
    while not isExhausted:
        optimizer.zero_grad()
        images_curr, normals_curr, masks_curr, isExhausted = loader.batch_train()
        normals_curr = normals_curr.long()
        if opts.cuda:
            images_curr = images_curr.cuda()
            normals_curr = normals_curr.cuda()
            masks_curr = masks_curr.cuda()

        images_curr = Variable(images_curr)
        normals_curr = Variable(normals_curr)
        masks_curr = Variable(masks_curr)
        preds_curr = F.log_softmax(net(images_curr), dim=1)

        loss = torch.sum(criterion(preds_curr, normals_curr)*masks_curr) / (torch.sum(masks_curr)+1e-5)

        loss.backward()
        optimizer.step()

        if (iter_no + 1) % 10 == 0:
            print('===> [epoch: %4d, iter: %4d] Loss: %.4f'%(epoch, iter_no, loss.data[0]))
            writer.add_scalar('data/train_loss', loss.data[0], epoch*total_iters + iter_no)
        
        iter_no += 1

    unbinned_score, binned_score = evaluate(net, loader, 'valid')
    print('Epoch %d validation results: Unbinned score: %.3f , Binned score: %.3f'%(epoch, unbinned_score, binned_score))

    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), iter_no)
    writer.add_scalar('data/valid_unbinned_accuracy', unbinned_score, epoch)
    writer.add_scalar('data/valid_binned_accuracy', binned_score, epoch)
   
    if binned_score >= best_binned_score:
        torch.save(net.state_dict(), os.path.join(opts.save_dir, 'model_best.net'))

    torch.save(net.state_dict(), os.path.join(opts.save_dir, 'model_latest.net'))

writer.close()
