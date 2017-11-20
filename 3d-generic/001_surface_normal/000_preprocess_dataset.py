from scipy.io import loadmat
from PIL import Image

import subprocess as sp
import numpy as np
import argparse
import h5py
import os

np.random.seed(123)

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal', type=str, help='path to surface normals data root')
parser.add_argument('--save_path', default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal/data.h5', type=str)
parser.add_argument('--num_val', default=25, type=str)
opts = parser.parse_args()

depth_data_h5 = h5py.File(os.path.join(opts.root_dir, 'nyu_depth_v2_labeled.mat'), 'r')

# Initially, it is N x C x W x H
# Change to N x C x H x W
images = np.array(depth_data_h5['images']).transpose(0, 1, 3, 2)

splits_mat = loadmat(os.path.join(opts.root_dir, 'splits.mat'), squeeze_me=True)
train_ids = splits_mat['trainNdxs']-1
np.random.shuffle(train_ids)
valid_ids = np.array(train_ids[-opts.num_val:])
train_ids = np.array(train_ids[:-opts.num_val])
test_ids = splits_mat['testNdxs']-1

ids = {"train": train_ids, "valid": valid_ids, "test": test_ids}

normals_path = os.path.join(opts.root_dir, 'normals_gt/normals')
masks_path = os.path.join(opts.root_dir, 'normals_gt/masks')

normals_files = sp.check_output(['ls', normals_path]).split()
masks_files = sp.check_output(['ls', masks_path]).split()

normals_gt = np.zeros(images.shape, dtype=np.uint8)
masks_gt = np.zeros((images.shape[0], images.shape[2], images.shape[3]), dtype=np.uint8)

for i in range(len(normals_files)):

    normal_curr = np.array(Image.open(os.path.join(normals_path, normals_files[i]))).transpose(2, 0, 1)
    mask_curr = np.array(Image.open(os.path.join(masks_path, masks_files[i])))
    normals_gt[i] = normal_curr[:, :, :]
    masks_gt[i] = mask_curr[:, :]
    
h5_out = h5py.File(opts.save_path, 'w')

for split in ids:
    h5_out.create_dataset('%s/images'%(split), data=np.take(images, ids[split], axis=0))
    h5_out.create_dataset('%s/normals'%(split), data=np.take(normals_gt, ids[split], axis=0))
    h5_out.create_dataset('%s/masks'%(split), data=np.take(masks_gt, ids[split], axis=0))

h5_out.close()
