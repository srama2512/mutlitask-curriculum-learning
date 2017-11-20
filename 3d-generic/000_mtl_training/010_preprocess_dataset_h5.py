import os
import pdb
import json
import h5py
import argparse
import subprocess
import numpy as np
from PIL import Image
import random

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_root', default='dataset/train/', type=str, help='Path to dataset train root')
parser.add_argument('--output_h5', default='dataset/train/regTrain/prepro.h5', type=str, help='Output h5 path')
parser.add_argument('--output_json', default='dataset/train/regTrain/prepro.json', type=str, help='Output json path')

opts = parser.parse_args()

random.seed(123)

images_path = os.path.join(opts.dataset_root, 'regTrain', 'data')

"""
    Get the set of positive and negative labels for matching. 
    The labels for matching consist of positives (1) / negative (0) label
    and also the distance between targets (if we want a curriculum for negatives).
    Also, get the set of pose labels for positives. These consist of the 
    (heading, pitch, roll, x_translation, y_translation, z_translation, baseline angle)
    for introducing a curriculum.
"""

positive_labels = []
negative_labels = []

pose_labels = []

"""
    Get the positive and negative image pairs
"""
positive_pairs = []
negative_pairs = []

pos_txt_path = os.path.join(opts.dataset_root, 'regTrain', 'regpairs_positive_refined.txt')
neg_txt_path = os.path.join(opts.dataset_root, 'regTrain', 'regpairs_negative_refined.txt')
print('===> Loading positive labels')
with open(pos_txt_path) as fp:
    # Read line by line
    for jdx in fp:
        l_split = jdx.split()
        # Store image names in pairs
        positive_pairs.append(l_split[0:2])
        # Store match label and target distance 
        positive_labels.append([int(l_split[2]), float(l_split[-1])])
        # Store (heading, pitch, roll, x_translation, y_translation, z_translation, baseline angle)
        pose_labels.append([float(l) for l in l_split[3:-1]])

print('===> Loading negative labels')
with open(neg_txt_path) as fp:
    # Read line by line
    for jdx in fp:
        l_split = jdx.split()
        # Store image names in pairs
        negative_pairs.append(l_split[0:2])
        # Store match label and target distance 
        negative_labels.append([int(l_split[2]), float(l_split[-1])])

print('Total number of positives: %d'%(len(positive_pairs)))
print('Total number of negatives: %d'%(len(negative_pairs)))

"""
    The preprocessed dataset is stored into prepro.h5 and prepro.json files.
    prepro.json contains the dict_of_images.
    prepro.h5 contains:
        1. positive/labels
        2. negative/labels
        3. pose_labels
        4. positive/images_left  (uint8)
        5. positive/images_right (uint8)
        6. negative/images_left  (uint8)
        7. negative/images_right (uint8)
"""

print('===> Creating output h5 file: %s'%(opts.output_h5))
h5_out = h5py.File(opts.output_h5, 'w')
h5_out.create_dataset("positive/labels", data=positive_labels)
h5_out.create_dataset("negative/labels", data=negative_labels)
h5_out.create_dataset("pose_labels", data=pose_labels)
print('===> Successfully wrote labels to %s'%(opts.output_h5))

"""
    Read the images and save them into h5file for faster access while testing.

"""
N_images = len(positive_labels)
h5_images_positive_left = h5_out.create_dataset("positive/images_left", (N_images, 3, 101, 101), dtype="uint8")
h5_images_positive_right = h5_out.create_dataset("positive/images_right", (N_images, 3, 101, 101), dtype="uint8")

for idx, img_pairs in enumerate(positive_pairs):
    # Load image and reshape from 101x101x3 to 3x101x101
    img_curr_left = np.transpose(np.array(Image.open(os.path.join(images_path, img_pairs[0]))), (2, 0, 1))
    img_curr_right = np.transpose(np.array(Image.open(os.path.join(images_path, img_pairs[1]))), (2, 0, 1))
    h5_images_positive_left[idx] = img_curr_left
    h5_images_positive_right[idx] = img_curr_right

    if (idx +1)%1000 == 0:
        print("Written out [%d/%d] images"%(idx+1, N_images))

print('\nWritten Positive images!\n')

N_images = len(negative_labels)
h5_images_negative_left = h5_out.create_dataset("negative/images_left", (N_images, 3, 101, 101), dtype="uint8")
h5_images_negative_right = h5_out.create_dataset("negative/images_right", (N_images, 3, 101, 101), dtype="uint8")

for idx, img_pairs in enumerate(negative_pairs):
    # Load image and reshape from 101x101x3 to 3x101x101
    img_curr_left = np.transpose(np.array(Image.open(os.path.join(images_path, img_pairs[0]))), (2, 0, 1))
    img_curr_right = np.transpose(np.array(Image.open(os.path.join(images_path, img_pairs[1]))), (2, 0, 1))
    h5_images_negative_left[idx] = img_curr_left
    h5_images_negative_right[idx] = img_curr_right

print('===> Finished writing images!')
h5_out.close()
