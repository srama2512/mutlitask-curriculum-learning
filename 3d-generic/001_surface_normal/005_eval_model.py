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
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', type=str2bool, default=True)
parser.add_argument('--model_path', type=str, required=True)

opts = parser.parse_args()

# Add misc to path
sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))
from Models import ModelSurfaceNormal
from DataLoaderSN import DataLoader

# Create DataLoader
loader = DataLoader(opts)

# Create model
net = ModelSurfaceNormal()

# Load model
chkpt = torch.load(opts.model_path)
net.load_state_dict(chkpt)

if opts.cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')
if opts.cuda:
    net = net.cuda()

def evaluate(net, loader, split):

    net.eval()
    isExhausted = False
    predicted_classes_all = []
    predicted_class_probs_all = []
    true_classes_all = []
    masks_all = []
    true_normals_all = []

    while not isExhausted:
        images_curr, normals_curr, masks_curr, true_normals_curr, isExhausted = loader.batch_test(split)
        if opts.cuda:
            images_curr = images_curr.cuda()
            normals_curr = normals_curr.cuda()
            masks_Curr = masks_curr.cuda()

        images_curr = Variable(images_curr, requires_grad=True)
        normals_curr = Variable(normals_curr, requires_grad=True)
        masks_curr = Variable(masks_curr, requires_grad=True)
        preds_curr = F.log_softmax(net(images_curr), dim=1)
        
        true_classes = normals_curr.cpu().data.numpy()
        predicted_class_probs = preds_curr.cpu().data.numpy()
        predicted_classes = np.argmax(preds_curr.cpu().data.numpy(), axis=1)
        
        predicted_class_probs_all.append(predicted_class_probs)
        true_classes_all.append(true_classes)
        predicted_classes_all.append(predicted_classes)
        masks_all.append(masks_curr.cpu().data.numpy())
        #true_normals_all.append(true_normals_curr)

    predicted_classes_all = np.concatenate(predicted_classes_all, axis=0)
    true_classes_all = np.concatenate(true_classes_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)
    #true_normals_all = np.concatenate(true_normals_all, axis=0)
    predicted_class_probs_all = np.concatenate(predicted_class_probs_all, axis=0)

    unbinned_score, binned_score = get_report(predicted_classes_all, true_classes_all, masks_all, 20) 
    net.train()
    
    return unbinned_score, binned_score

unbinned_score, binned_score = evaluate(net, loader, 'valid')
print('Validation results: Unbinned score: %.3f , Binned score: %.3f'%(unbinned_score, binned_score))
unbinned_score, binned_score = evaluate(net, loader, 'test')
print('Test results: Unbinned score: %.3f , Binned score: %.3f'%(unbinned_score, binned_score))

