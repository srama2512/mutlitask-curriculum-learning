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
parser.add_argument('--subdirs', default='2-3-4-9-12-15-17-20', type=str, help='Specify as <sub1>-<sub2>-...')
parser.add_argument('--output_h5', default='dataset/train/regTrain/prepro.h5', type=str, help='Output h5 path')
parser.add_argument('--output_json', default='dataset/train/regTrain/prepro.json', type=str, help='Output json path')

opts = parser.parse_args()

random.seed(123)

images_path = os.path.join(opts.dataset_root, 'regTrain', 'data')
subdirs = ['%.4d'%(int(i)) for i in opts.subdirs.split('-')]

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
    Get a dict of all the images needed for the dataset. Assign them an index and
    used the indices for creating image pairs.
"""
print('===> Reading image names from %s'%(images_path))
list_of_images = subprocess.check_output(['ls', images_path+'/']).split()
print('Total number of images: %d'%(len(list_of_images)))
dict_of_images = {v: k for k, v in enumerate(list_of_images)}

"""
    Get the positive and negative image pairs
"""
positive_pairs = []
negative_pairs = []

for idx, subdir in enumerate(subdirs):
    pos_txt_path = os.path.join(opts.dataset_root, 'regTrain', 'regpairs_positive_%s.txt'%(subdir))
    neg_txt_path = os.path.join(opts.dataset_root, 'regTrain', 'regpairs_negative_%s.txt'%(subdir))
    print('===> Loading positive labels for %s subset'%(subdir))
    with open(pos_txt_path) as fp:
        # Read line by line
        for jdx in fp:
            l_split = jdx.split()
            # Store image indices in pairs
            positive_pairs.append([dict_of_images[img] for img in l_split[0:2]])
            # Store match label and target distance 
            positive_labels.append([int(l_split[2]), float(l_split[-1])])
            # Store (heading, pitch, roll, x_translation, y_translation, z_translation, baseline angle)
            pose_labels.append([float(l) for l in l_split[3:-1]])

    print('===> Loading negative labels for %s subset'%(subdir))
    with open(neg_txt_path) as fp:
        # Read line by line
        for jdx in fp:
            l_split = jdx.split()
            # Store image indices in pairs
            negative_pairs.append([dict_of_images[img] for img in l_split[0:2]])
            # Store match label and target distance 
            negative_labels.append([int(l_split[2]), float(l_split[-1])])


print('Total number of positives: %d'%(len(positive_pairs)))
print('Total number of negatives: %d'%(len(negative_pairs)))

# Shuffle the positive and negative data samples 
p_zipped = zip(positive_pairs, positive_labels, pose_labels)
n_zipped = zip(negative_pairs, negative_labels)

random.shuffle(p_zipped)
random.shuffle(n_zipped)

positive_pairs_shuffled, positive_labels_shuffled, pose_labels_shuffled = zip(*p_zipped)
negative_pairs_shuffled, negative_labels_shuffled = zip(*n_zipped)

"""
    The preprocessed dataset is stored into prepro.h5 and prepro.json files.
    prepro.json contains the dict_of_images.
    prepro.h5 contains:
        1. positive_pairs
        2. negative_pairs
        3. positive_labels
        4. negative_labels
        5. pose_labels
        6. images (uint8)
"""

print('===> Creating output h5 file: %s'%(opts.output_h5))
h5_out = h5py.File(opts.output_h5, 'w')
h5_out.create_dataset("positive_pairs", data=positive_pairs_shuffled)
h5_out.create_dataset("negative_pairs", data=negative_pairs_shuffled)
h5_out.create_dataset("positive_labels", data=positive_labels_shuffled)
h5_out.create_dataset("negative_labels", data=negative_labels_shuffled)
h5_out.create_dataset("pose_labels", data=pose_labels_shuffled)
print('===> Successfully wrote labels to %s'%(opts.output_h5))

"""
    Read the images and save them into h5file for faster access while training.

"""
N_images = len(list_of_images)
h5_images = h5_out.create_dataset("images", (N_images, 3, 101, 101), dtype="uint8")
for idx, img in enumerate(list_of_images):
    # Load image and reshape from 101x101x3 to 3x101x101
    img_curr = np.transpose(np.array(Image.open(os.path.join(images_path, img))), (2, 0, 1))
    h5_images[idx] = img_curr
    if (idx +1)%1000 == 0:
        print("Written out [%d/%d] images"%(idx+1, N_images))

print('===> Finished writing images!')
h5_out.close()
json.dump(dict_of_images, open(opts.output_json, 'w'))
print('===> Finished writing %s'%(opts.output_json))
