from utils import average_angular_error, average_translation_error
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import matplotlib.pyplot as plt
from argparse import Namespace

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

from Models import ModelPose
net = ModelPose()

net.load_state_dict(torch.load(opts.load_model))

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opts.cuda:
    net = net.cuda()

net.eval()

# Compute pose estimation results
num_samples = test_data_h5['pose_labels'].shape[0]

if num_samples % opts.batch_size == 0:
    num_batches = num_samples // opts.batch_size
else:
    num_batches = num_samples // opts.batch_size + 1

pose_mean = [ 19.70189285,   0.59445256,  0,  7.25698948, -11.20506763,   0.05411328]
pose_std = [ 74.65267944,   6.41094828,  0, 29.34560776, 24.62372398,   2.63554645]
img_mean = [ 123.675,  116.28,  103.53 ]
img_std = [ 58.395,  57.12,   57.375]

pose_predictions = np.zeros((num_samples, 6))
pose_labels = test_data_h5['pose_labels'][:, :6]

# JSON dict to dump
json_out_dict = {}

start = 0

for i in range(num_batches):
    end = min(start + opts.batch_size, num_samples)
    curr_batch_size = end - start
    imgs_left = test_data_h5['positive']['images_left'][start:end].astype(np.float)
    imgs_right = test_data_h5['positive']['images_right'][start:end].astype(np.float)
    
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
    pose_predictions[start:end, 0:2] = pred[:, 0:2]
    pose_predictions[start:end, 3:] = pred[:, 2:]

    start += curr_batch_size

for j in range(len(pose_mean)):
    pose_predictions[:, j] *= pose_std[j]
    pose_predictions[:, j] += pose_mean[j]

aae = average_angular_error(pose_predictions[:, :3], pose_labels[:, :3])
ate = average_translation_error(pose_predictions[:, 3:], pose_labels[:, 3:])
aes = average_angular_error(pose_predictions[:, :3], pose_labels[:, :3], average=False)
tes = average_translation_error(pose_predictions[:, 3:], pose_labels[:, 3:], average=False)

json_out_dict['aae'] = aae
json_out_dict['ate'] = ate
json_out_dict['aes'] = list(aes)
json_out_dict['tes'] = list(tes)
json_out_dict['key'] = opts.key

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
print('ATE: %.3f    AAE: %.3f'%(ate, aae))

if not os.path.isdir(opts.result_path):
    sp.call('mkdir %s'%(opts.result_path), shell=True)

json.dump(json_out_dict, open(os.path.join(opts.result_path, '%s.json'%(opts.key)), 'w'))
