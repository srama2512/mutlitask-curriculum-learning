from PIL import Image

from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
import subprocess as sp
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
parser.add_argument('--num_clusters', type=int, default=20, help='number of words in codebook')

opts = parser.parse_args()

data_h5 = h5py.File(opts.data_path, 'r')

train_normals = np.array(data_h5['train/normals'][:200]).transpose(0, 2, 3, 1).reshape(-1, 3).astype(np.float)/255.0 * 2 - 1
train_masks = np.array(data_h5['train/masks'][:200]).reshape(-1) > 0
train_normals_valid = train_normals[train_masks, :]
train_normals_norm = np.linalg.norm(train_normals_valid, axis=1)
train_normals_valid /= train_normals_norm[:, np.newaxis]

# Initialize Clustering object
if not os.path.isfile('temp_clusters.json'):
    kmeans_object = KMeans(n_clusters=20, max_iter=200, n_init=5, verbose=1, n_jobs=8)
    clusters = kmeans_object.fit(train_normals_valid)
    cluster_centers = clusters.cluster_centers_
    json.dump(clusters.cluster_centers_.tolist(), open('temp_clusters.json', 'w'))
else:
    cluster_centers = np.array(json.load(open('temp_clusters.json', 'r')))
