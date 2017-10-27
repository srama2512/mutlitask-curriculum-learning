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
import random

from PIL import Image, ImageDraw, ImageFont
from utils import *

random.seed(123)

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

#for base_dir in ['0002/', '0003/', '0004/', '0009/', '0012/']:
for base_dir in ['0014/', '0015/', '0017/', '0020/']:
    
    regpairs_negative = open(os.path.join(opts.save_dir, 'regTrain', 'regpairs_negative_%s.txt'%(base_dir[:-1])), 'w')
    targets = create_target_cache(opts.dataset_dir, base_dir) 
    targetIDs = targets.keys()

    print('\nProcessing data subset: %s'%(base_dir))
    count_targets = 0
    for targetIDX, targetID in enumerate(targetIDs):
    
        count_targets += 1
        targetCurr = targets[targetID]
        nViews = len(targetCurr['views'])

        for i in range(nViews):
            view_i = targetCurr['views'][i]
            
            # If the SSI score of view_i >= 0.15
            if float(view_i['alignData'][32]) >= 0.15:
                
                # Extract the name of the current patch
                name_i = view_i['imagePath'].split('/')[1]
                load_path_i = os.path.join(opts.dataset_dir, view_i['imagePath'])
                save_path_i = os.path.join(opts.save_dir, 'regTrain', 'data', name_i)
                
                # Save this image only if it has not been saved before
                if not os.path.isfile(save_path_i):
                
                    img_curr = Image.open(load_path_i)
            
                    # Crop a 192x192 image centered around the aligned center
                    l1 = int(float(view_i['alignData'][1]) - 96)
                    t1 = int(float(view_i['alignData'][2]) - 96)
                    r1 = l1 + 192 - 1
                    b1 = t1 + 192 - 1
                    box1 = (l1, t1, r1, b1)
                    img_curr = img_curr.crop(box1)
                    # Resize image to 101x101 
                    img_curr = img_curr.resize([101, 101])
                    img_curr.save(save_path_i)

                # Randomly select 3 other targets
                for j in range(3):
                    jIDX = random.randint(0, len(targetIDs)-1)
                    # Ensure same nearby targets is not selected
                    if abs(jIDX - targetIDX) <= 200 :
                        jIDX = (jIDX + 200)%len(targetIDs)
                        
                    jID = targetIDs[jIDX]
                    # Compute the distance between targets
                    dist_targets = haversine_distance(targets[targetID]['targetCoord'], \
                                                      targets[jID]['targetCoord'])

                    # The distance between the targets must be atleast 50m
                    #while dist_targets < 50:
                    #    jIDX = random.randint(0, len(targetIDs)-1)
                    #    if jIDX == targetIDX:
                    #        jIDX += 1
                    #    jID = targetIDs[jIDX]
                        # Compute the distance between targets
                    #    dist_targets = haversine_distance(targets[targetID]['targetCoord'], \
                    #                                  targets[jID]['targetCoord'])
                    # Always select the 1st view of a target
                    view_j = targets[jID]['views'][0]
                    name_j = view_j['imagePath'].split('/')[1]
                    load_path_j = os.path.join(opts.dataset_dir, view_j['imagePath'])
                    save_path_j = os.path.join(opts.save_dir, 'regTrain', 'data', name_j)
                    
                    # Save this image only if it has not been saved before
                    if not os.path.isfile(save_path_j):
                    
                        img_curr = Image.open(load_path_j)
                        # Crop a 192x192 image centered around the aligned center
                        l1 = int(float(view_j['alignData'][1]) - 96)
                        t1 = int(float(view_j['alignData'][2]) - 96)
                        r1 = l1 + 192 - 1
                        b1 = t1 + 192 - 1
                        box1 = (l1, t1, r1, b1)
                        img_curr = img_curr.crop(box1)
                        # Resize image to 101x101 
                        img_curr = img_curr.resize([101, 101])
                        img_curr.save(save_path_j)

                    # Since these are not matches, set the relative pose, base_angle and translation
                    # to a degenerate value
                    rel_pose = [0, 0, 0]
                    rel_trans = [0, 0, 0] 
                    b_angle = 0 

                    # Write out the annotation
                    regpairs_negative.write('%s %s %s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n'\
                                          %(name_i, name_j, '0', rel_pose[0],\
                                          rel_pose[1], rel_pose[2], rel_trans[0],\
                                          rel_trans[1], rel_trans[2], b_angle, dist_targets))

        if count_targets % 50 == 0:
            sys.stdout.write('Finished target [%d/%d]\r'%(count_targets, len(targets)))
            sys.stdout.flush()
    
    print('\n')
    regpairs_negative.close()
