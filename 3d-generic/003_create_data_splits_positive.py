"""
This script creates the training data splits for both the pose
regression and wide baseline stereo matching tasks. 
"""

"""
Import the necessary libraries here
"""
import numpy as np
import subprocess
import argparse
import pickle
import math
import pdb
import sys
import os

from PIL import Image, ImageDraw, ImageFont
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='dataset/train/', type=str, \
                    help='Root directory of dataset')
parser.add_argument('--save_dir', default='dataset/train/', type=str, \
                    help='Path to save the pose estimation and matching splits created')
opts = parser.parse_args()

subprocess.call('mkdir %s'%(os.path.join(opts.save_dir, 'regTrain')), shell=True)

# Create pose regression and matching data
subprocess.call('mkdir %s'%(os.path.join(opts.save_dir, 'regTrain', 'data')), shell=True)
with open(os.path.join(opts.save_dir, 'regTrain', 'format.txt'), 'w') as fopen:
    fopen.write('<imgA> <imgB> <match or no match> <relative heading degrees> \
                 <relative pitch degrees> <relative roll degrees> <relative longitual translation meters> \
                 <relative lateral translation meters> <relative height meters> \
                 <baseline angle in degrees> <distance between targets>')

for base_dir in ['0002/', '0003/', '0004/', '0009/', '0012/']:
#for base_dir in ['0014/', '0015/', '0017/', '0020/']:
    
    regpairs_positive = open(os.path.join(opts.save_dir, 'regTrain', 'regpairs_positive_%s.txt'%(base_dir[:-1])), 'w')
    targets = create_target_cache(opts.dataset_dir, base_dir) 
    print('\nProcessing data subset: %s'%(base_dir))
    count_targets = 0
    for targetID in targets:
        count_targets += 1
        targetCurr = targets[targetID]
        nViews = len(targetCurr['views'])
        views_to_write = set()
        for i in range(nViews-1):
            view_i = targetCurr['views'][i]
            # If the SSI score of view_i >= 0.15
            if float(view_i['alignData'][32]) >= 0.15:
                for j in range(i+1, nViews):
                    view_j = targetCurr['views'][j]

                    # If the SSI score of view_j >= 0.15
                    if float(view_j['alignData'][32]) >= 0.15:
                        b_angle = baseline_angle_1(view_i['cameraCoord'], view_j['cameraCoord'], \
                                                   targetCurr['targetCoord'])
                        # Extract the name of the images
                        name_i = view_i['imagePath'].split('/')[1]
                        name_j = view_j['imagePath'].split('/')[1]
                   
                        # These view images have to be written out to dataset
                        views_to_write.add(i)
                        views_to_write.add(j)

                        # Compute relative pose (view_j - view_i)
                        rel_pose = relative_rotation(view_i['cameraPose'], view_j['cameraPose'])

                        # Compute relative translation (view_j - view_i)
                        rel_trans = relative_translation(view_i['cameraCoord'], view_j['cameraCoord'])
                        
                        # Write out the annotation
                        regpairs_positive.write('%s %s %s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %s\n'\
                                              %(name_i, name_j, '1', rel_pose[0],\
                                              rel_pose[1], rel_pose[2], rel_trans[0],\
                                              rel_trans[1], rel_trans[2], b_angle, str(0)))

        for i in views_to_write:
            img_path = targetCurr['views'][i]['imagePath']
            load_path = os.path.join(opts.dataset_dir, img_path)
            save_path = os.path.join(opts.save_dir, 'regTrain', 'data', img_path.split('/')[1])
            img_curr = Image.open(load_path)
            
            # Crop a 192x192 image centered around the aligned center
            l1 = int(float(targetCurr['views'][i]['alignData'][1]) - 96)
            t1 = int(float(targetCurr['views'][i]['alignData'][2]) - 96)
            r1 = l1 + 192 - 1
            b1 = t1 + 192 - 1
            box1 = (l1, t1, r1, b1)
            img_curr = img_curr.crop(box1)
            # Resize image to 101x101 
            img_curr = img_curr.resize([101, 101])
            img_curr.save(save_path)
        
        if count_targets % 100 == 0:
            sys.stdout.write('Finished target [%d/%d]\r'%(count_targets, len(targets)))
            sys.stdout.flush()
    
    print('\n')
    regpairs_positive.close()
