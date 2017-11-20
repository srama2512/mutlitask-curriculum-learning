from scipy.optimize import nnls
from scipy.misc import imresize
import subprocess as sp
import progressbar
import numpy as np
import argparse
import h5py
import json
import pdb
import os

np.random.seed(123)

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal', type=str, help='path to surface normals data root')
parser.add_argument('--data_path', default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal/data.h5', type=str, help='path to preprocessed dataset')
parser.add_argument('--save_path', default='/scratch/cluster/srama/Fall2017-CS381V/project/3d-generic/dataset/surface_normal/normals.h5')

opts = parser.parse_args()

cluster_centers = np.array(json.load(open('temp_clusters.json', 'r')))
delaunay_vertices = np.array(json.load(open('delaunay_vertices.json', 'r')))

cluster_centers = cluster_centers.transpose()
# Get cluster centers corresponding to each triangle
Bs = []
for t in range(delaunay_vertices.shape[0]):
    B = np.take(cluster_centers, delaunay_vertices[t], axis=1)
    Bs.append(B)

data_h5 = h5py.File(opts.data_path, 'r')
h5_out = h5py.File(opts.save_path, 'w')

for split in ['train', 'valid', 'test']:
    split_normals = np.zeros((data_h5['%s/normals'%(split)].shape[0], 20, 20, 3), dtype=np.uint8)
    split_masks = np.zeros((data_h5['%s/masks'%(split)].shape[0], 20, 20), dtype=np.uint8)

    # N x H x W x C format for split_normals
    bar = progressbar.ProgressBar()
    print('Processing %s split'%(split))
    print('===> Reading normals and masks')
    for i in bar(range(split_normals.shape[0])):
        split_normals[i, :, :, :] = imresize(data_h5['%s/normals'%(split)][i].transpose(1, 2, 0), size=(20, 20), interp='nearest')
        split_masks[i, :, :] = imresize(data_h5['%s/masks'%(split)][i], size=(20, 20), interp='nearest')

    split_normals = 2.0*split_normals.astype(np.float32)/255.0 - 1.0
    # Classes assigned to each normal
    split_normals_classes = np.zeros((split_normals.shape[0], split_normals.shape[1], split_normals.shape[2]), dtype=np.uint8)

    bar = progressbar.ProgressBar()
    # For every pixel of every image, perform class assignment
    print('===> Assigning class labels')
    for i in bar(range(split_normals.shape[0])):
        for j in range(split_normals.shape[1]):
            for k in range(split_normals.shape[2]):
                if split_masks[i, j, k] > 0:
                    n_curr = split_normals[i, j, k, :]
                    min_error = 1000
                    t_min_error = 0
                    coeffs_min_error = None
                    for t in range(delaunay_vertices.shape[0]):
                        coeffs_curr, error = nnls(Bs[t], n_curr)
                        if error < min_error:
                            min_error = error
                            t_min_error = t
                            coeffs_min_error = coeffs_curr
                    split_normals_classes[i, j, k] = delaunay_vertices[t_min_error][np.argmax(coeffs_min_error)]

    h5_out.create_dataset('%s/normals'%(split), data=split_normals_classes)
    h5_out.create_dataset('%s/masks'%(split), data=split_masks)

h5_out.close()
