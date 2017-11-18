from utils import auc_score, average_angular_error, average_translation_error
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import matplotlib.pyplot as plt
from argparse import Namespace

import torch.nn.functional as F
import torch.optim as optim
import subprocess as sp
import torch.nn as nn
from PIL import Image
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
parser.add_argument('--images_path', type=str, default='dataset/vanishing_point/YorkUrbanDB/images', help='path to directory containing images')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--cuda', type=str2bool, default=True)
parser.add_argument('--load_model', type=str, default='', help='saved model path')
parser.add_argument('--result_path', type=str, default='', help='path to store results')
parser.add_argument('--key', type=str, default='default', help='key to identify json')

opts = parser.parse_args()

print('Evaluating %s'%(opts.key))

# Add misc to path
sys.path.append(os.path.join(os.path.abspath(''), 'misc/'))

from Models import ModelJoint
net = ModelJoint()

net.load_state_dict(torch.load(opts.load_model))

if opts.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if opts.cuda:
    net = net.cuda()

net.eval()

# Get embeddings and write to tensorboard
writer = SummaryWriter(log_dir = opts.result_path)

images_embedding = []
images_raw = []

image_files = sp.check_output(['ls', opts.images_path]).split()
num_samples = len(image_files)

if num_samples % opts.batch_size == 0:
    num_batches = num_samples // opts.batch_size
else:
    num_batches = num_samples // opts.batch_size + 1

img_mean = [ 123.675,  116.28,  103.53 ]
img_std = [ 58.395,  57.12,   57.375]

start = 0

imgs_all = np.zeros((num_samples, 3, 101, 101), dtype=np.uint8)

for i in range(len(image_files)):
    img_temp = Image.open(os.path.join(opts.images_path, image_files[i]))
    img_temp = img_temp.resize([101, 101])
    imgs_all[i] = np.transpose(np.array(img_temp), (2, 0, 1))

np.random.shuffle(imgs_all)

for i in range(5):#num_batches):
    end = min(start + opts.batch_size, num_samples)
    curr_batch_size = end - start
    # LEFT and RIGHT have no meaning here
    imgs = imgs_all[start:end].astype(np.float)
    
    for j in range(3):
        imgs[:, j, :, :] -= img_mean[j]
        imgs[:, j, :, :] /= (img_std[j] + 1e-5)

    imgs = Variable(torch.Tensor(imgs))
    if opts.cuda:
        imgs = imgs.cuda()

    feats = net.forward_feature(imgs)
    imgs_raw = imgs_all[start:end]

    images_raw.append(imgs_raw)
    images_embedding.append(feats.cpu().data.numpy())

    start += curr_batch_size

images_embedding = torch.Tensor(np.vstack(images_embedding)).float()
images_raw = torch.ByteTensor(np.vstack(images_raw))
writer.add_embedding(images_embedding, label_img=images_raw.float()/255)

writer.close()

"""
print('ATE: %.3f    AAE: %.3f   AUC: %.3f'%(ate, aae, auc))
json_out_dict['auc'] = auc
json_out_dict['tpr'] = list(tpr)
json_out_dict['fpr'] = list(fpr)
json_out_dict['thresh'] = list(thresh)

if not os.path.isdir(opts.result_path):
    sp.call('mkdir %s'%(opts.result_path), shell=True)

json.dump(json_out_dict, open(os.path.join(opts.result_path, '%s.json'%(opts.key)), 'w'))
"""
