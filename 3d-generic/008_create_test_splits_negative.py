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
import random
import math
import pdb
import sys
import os

from PIL import Image, ImageDraw, ImageFont
from utils import *

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='dataset/test/', type=str, \
                    help='Root directory of dataset')
parser.add_argument('--save_dir', default='dataset/test/', type=str, \
                    help='Path to save the pose estimation and matching splits created')
opts = parser.parse_args()

subprocess.call('mkdir %s'%(os.path.join(opts.save_dir, 'regTest')), shell=True)

# Create pose regression and matching data
num_positive = int(subprocess.check_output('ls %s | wc -l'%(os.path.join(opts.save_dir, 'regTest', 'data')), shell=True)) // 2

regpairs_negative = open(os.path.join(opts.save_dir, 'regTest', 'regpairs_negative.txt'), 'a')

# This is used to decide the save path of images
count_matches = num_positive + 1

for base_dir in ['0096/']:
    
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
            
            # Extract the name of the current patch
            name_i = view_i['imagePath'].split('/')[1]
            load_path_i = os.path.join(opts.dataset_dir, view_i['imagePath'])
            save_path_i = os.path.join(opts.save_dir, 'regTest', 'data', name_i)
            img_i_present = True
            try:
                img_open_i = Image.open(load_path_i)
            except IOError:
                img_i_present = False
                print('Image %s not present!'%(os.path.join(opts.dataset_dir, view_i['imagePath'])))
            
            # If the SSI score of view_i >= 0.20
            if float(view_i['alignData'][32]) >= 0.20 and img_i_present:
 
                img_curr = Image.open(load_path_i)
        
                # Randomly select 1 other targets
                jIDX = random.randint(0, len(targetIDs)-1)
                # Ensure same nearby targets is not selected
                if abs(jIDX - targetIDX) <= 200 :
                    jIDX = (jIDX + 200)%len(targetIDs)
                    
                jID = targetIDs[jIDX]
                # Compute the distance between targets
                dist_targets = haversine_distance(targets[targetID]['targetCoord'], \
                                                  targets[jID]['targetCoord'])

                # Always select the 1st view of a target
                view_j = targets[jID]['views'][0]
                name_j = view_j['imagePath'].split('/')[1]
                
                load_path_j = os.path.join(opts.dataset_dir, view_j['imagePath'])
                img_j_present = True
                try:
                    img_open_j = Image.open(load_path_j)
                except IOError:
                    img_j_present = False

                if not img_j_present:
                    continue
                
                save_name_i = '%.7d_i.jpg'%(count_matches)
                save_name_j = '%.7d_p.jpg'%(count_matches)
                count_matches += 1

                save_path_i = os.path.join(opts.save_dir, 'regTest', 'data', save_name_i)
                save_path_j = os.path.join(opts.save_dir, 'regTest', 'data', save_name_j)
                
                for data_pairs in zip([save_path_i, save_path_j], [view_i['alignData'], view_j['alignData']], [img_open_i, img_open_j]):
                    
                    save_path_curr = data_pairs[0]
                    align_data = data_pairs[1]
                    img_curr = data_pairs[2]
                    # Crop a 192x192 image centered around the aligned center
                    l1 = int(float(align_data[1]) - 96)
                    t1 = int(float(align_data[2]) - 96)
                    r1 = l1 + 192 - 1
                    b1 = t1 + 192 - 1
                    box1 = (l1, t1, r1, b1)
                    img_curr = img_curr.crop(box1)
                    # Resize image to 101x101 
                    img_curr = img_curr.resize([101, 101])
                    img_curr.save(save_path_curr)

                # Since these are not matches, set the relative pose, base_angle and translation
                # to a degenerate value
                rel_pose = [0, 0, 0]
                rel_trans = [0, 0, 0] 
                b_angle = 0 

                # Write out the annotation
                regpairs_negative.write('%s %s %s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n'\
                                      %(save_name_i, save_name_j, '0', rel_pose[0],\
                                      rel_pose[1], rel_pose[2], rel_trans[0],\
                                      rel_trans[1], rel_trans[2], b_angle, dist_targets))

        if count_targets % 50 == 0:
            sys.stdout.write('Finished target [%d/%d]\n'%(count_targets, len(targets)))
            sys.stdout.flush()
    
    print('\n')

regpairs_negative.close()
