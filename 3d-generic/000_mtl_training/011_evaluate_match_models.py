from tensorboardX import SummaryWriter
from torch.autograd import Variable
import matplotlib.pyplot as plt
from argparse import Namespace
from utils import auc_score

import torch.nn.functional as F
import torch.optim as optim
import subprocess as sp
import torch.nn as nn
import numpy as np
import torchvision
import argparse
import torch
import h5py
import json
import sys
import pdb
import os

np.random.seed(123)

def str2bool(s):
    return s.lower() in ('yes', 'true', 't', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--h5_path', type=str, default='../dataset/test/regTest/prepro.h5', help='path to preprocessed h5 file')
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--cuda', type=str2bool, default=True)
parser.add_argument('--load_model', type=str, default='', help='saved model path')
parser.add_argument('--result_path', type=str, default='', help='path to store results')
parser.add_argument('--key', type=str, default='default', help='key to identify json')

opts = parser.parse_args()

print('Evaluating %s'%(opts.key))
test_data_h5 = h5py.File(opts.h5_path, 'r')

# Add misc to path
sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))

from Models import ModelMatch
net = ModelMatch()

net.load_state_dict(torch.load(opts.load_model))

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opts.cuda:
    net = net.cuda()

net.eval()

img_mean = [ 123.675,  116.28,  103.53 ]
img_std = [ 58.395,  57.12,   57.375]

# JSON dict to dump
json_out_dict = {}

# Compute matching results
num_samples = test_data_h5['positive/labels'].shape[0] + test_data_h5['negative/labels'].shape[0]
match_images_left = np.concatenate([np.array(test_data_h5['positive/images_left']), \
                                   np.array(test_data_h5['negative/images_left'])], 0)
match_images_right = np.concatenate([np.array(test_data_h5['positive/images_right']), \
                                    np.array(test_data_h5['negative/images_right'])], 0)

if num_samples % opts.batch_size == 0:
    num_batches = num_samples // opts.batch_size
else:
    num_batches = num_samples // opts.batch_size + 1

match_predictions = np.zeros((num_samples))
match_labels = np.concatenate([np.array(test_data_h5['positive/labels']), \
            np.array(test_data_h5['negative/labels'])], 0)[:, 0]

start = 0
for i in range(num_batches):
    end = min(start + opts.batch_size, num_samples)
    curr_batch_size = end - start
    imgs_left = match_images_left[start:end].astype(np.float)
    imgs_right = match_images_right[start:end].astype(np.float)
    for j in range(3):
        imgs_left[:, j, :, :] -= img_mean[j]
        imgs_left[:, j, :, :] /= (img_std[j] + 1e-5)
        imgs_right[:, j, :, :] -= img_mean[j]
        imgs_right[:, j, :, :] /= (img_std[j] + 1e-5)
    
    imgs_left = Variable(torch.Tensor(imgs_left))
    imgs_right = Variable(torch.Tensor(imgs_right))
    if opts.cuda:
        imgs_left = imgs_left.cuda()
        imgs_right = imgs_right.cuda()

    pred = net.forward(imgs_left, imgs_right).cpu().data.numpy()
    match_predictions[start:end] = pred[:, 0]
    
    start += curr_batch_size
    
auc, tpr, fpr, thresh = auc_score(match_predictions, match_labels, get_roc=True )

"""
# Get embeddings and write to tensorboard
writer = SummaryWriter(log_dir = opts.result_path)

images_embedding = []
images_raw = []

num_samples = test_data_h5['pose_labels'].shape[0]

if num_samples % opts.batch_size == 0:
    num_batches = num_samples // opts.batch_size
else:
    num_batches = num_samples // opts.batch_size + 1

img_mean = [ 123.675,  116.28,  103.53 ]
img_std = [ 58.395,  57.12,   57.375]

start = 0

imgs_all_left = np.array(test_data_h5['positive']['images_left'])
imgs_all_right = np.array(test_data_h5['positive']['images_right'])
#np.random.shuffle(imgs_all_left)
#np.random.shuffle(imgs_all_right)

for i in range(10):#num_batches):
    end = min(start + opts.batch_size, num_samples)
    curr_batch_size = end - start
    # LEFT and RIGHT have no meaning here
    imgs_left = imgs_all_left[start:end].astype(np.float)
    imgs_right = imgs_all_right[start:end].astype(np.float)
    
    for j in range(3):
        imgs_left[:, j, :, :] -= img_mean[j]
        imgs_left[:, j, :, :] /= (img_std[j] + 1e-5)
        imgs_right[:, j, :, :] -= img_mean[j]
        imgs_right[:, j, :, :] /= (img_std[j] + 1e-5)

    imgs_left = Variable(torch.Tensor(imgs_left))
    imgs_right = Variable(torch.Tensor(imgs_right))
    if opts.cuda:
        imgs_left = imgs_left.cuda()
        imgs_right = imgs_right.cuda()

    feats_left = net.forward_feature(imgs_left)
    feats_right = net.forward_feature(imgs_right)
    imgs_raw_left = test_data_h5['positive']['images_left'][start:end]
    imgs_raw_right = test_data_h5['positive']['images_right'][start:end]

    images_raw.append(imgs_raw_left)
    images_raw.append(imgs_raw_right)
    images_embedding.append(feats_left.cpu().data.numpy())
    images_embedding.append(feats_right.cpu().data.numpy())

    start += curr_batch_size

images_embedding = torch.Tensor(np.vstack(images_embedding)).float()
images_raw = torch.ByteTensor(np.vstack(images_raw))
writer.add_embedding(images_embedding, label_img=images_raw.float()/255)

writer.close()
"""

print('AUC: %.3f'%(auc))
json_out_dict['auc'] = auc
json_out_dict['tpr'] = list(tpr)
json_out_dict['fpr'] = list(fpr)
json_out_dict['thresh'] = list(thresh)

if not os.path.isdir(opts.result_path):
    sp.call('mkdir %s'%(opts.result_path), shell=True)

json.dump(json_out_dict, open(os.path.join(opts.result_path, '%s.json'%(opts.key)), 'w'))
