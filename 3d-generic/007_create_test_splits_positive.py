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
parser.add_argument('--dataset_dir', default='dataset/test/', type=str, \
                    help='Root directory of dataset')
parser.add_argument('--save_dir', default='dataset/test/', type=str, \
                    help='Path to save the pose estimation and matching splits created')
opts = parser.parse_args()

subprocess.call('mkdir %s'%(os.path.join(opts.save_dir, 'regTest')), shell=True)

# Create pose regression and matching data
subprocess.call('mkdir %s'%(os.path.join(opts.save_dir, 'regTest', 'data')), shell=True)
with open(os.path.join(opts.save_dir, 'regTest', 'format.txt'), 'w') as fopen:
    fopen.write('<imgA> <imgB> <match or no match> <relative heading degrees> \
                 <relative pitch degrees> <relative roll degrees> <relative longitual translation meters> \
                 <relative lateral translation meters> <relative height meters> \
                 <baseline angle in degrees> <distance between targets>')

# This is used to decide the save path of images
count_matches = 0
regpairs_positive = open(os.path.join(opts.save_dir, 'regTest', 'regpairs_positive.txt'), 'w')
for base_dir in ['0081/', '0096/']:
    
    targets = create_target_cache(opts.dataset_dir, base_dir) 
    print('\nProcessing data subset: %s'%(base_dir))
    count_targets = 0
    for targetID in targets:
        count_targets += 1
        targetCurr = targets[targetID]
        nViews = len(targetCurr['views'])

        # After one match is found, it will break out
        #found_match_for_target = False
        for i in range(nViews-1):
            #if found_match_for_target:
            #    break
            view_i = targetCurr['views'][i]
            # If the SSI score of view_i >= 0.2
            img_i_present = True
            try:
                img_open_i = Image.open(os.path.join(opts.dataset_dir, view_i['imagePath']))
            except IOError:
                img_i_present = False
                print('Image %s not present!'%(os.path.join(opts.dataset_dir, view_i['imagePath'])))

            if float(view_i['alignData'][32]) >= 0.2 and img_i_present:
                for j in range(i+1, nViews):
                    #if found_match_for_target:
                    #    break
                    view_j = targetCurr['views'][j]
                    img_j_present = True
                    try:
                        img_open_j = Image.open(os.path.join(opts.dataset_dir, view_j['imagePath']))
                    except IOError:
                        img_j_present = False
                    
                    # If the SSI score of view_j >= 0.2
                    if float(view_j['alignData'][32]) >= 0.2 and img_j_present:
                        b_angle = baseline_angle_1(view_i['cameraCoord'], view_j['cameraCoord'], \
                                                   targetCurr['targetCoord'])
                        # Extract the name of the images
                        name_i = view_i['imagePath'].split('/')[1]
                        name_j = view_j['imagePath'].split('/')[1]
                   
                        # Compute relative pose (view_j - view_i)
                        rel_pose = [v[1] - v[0] for v in zip(view_i['cameraPose'], view_j['cameraPose'])]
                        #relative_rotation(view_i['cameraPose'], view_j['cameraPose'])

                        # Compute relative translation (view_j - view_i)
                        rel_trans = relative_translation(view_i['cameraCoord'], view_j['cameraCoord']) 
                        
                        load_path_i = os.path.join(opts.dataset_dir, view_i['imagePath'])
                        load_path_j = os.path.join(opts.dataset_dir, view_j['imagePath'])
                        save_name_i = '%.7d_i.jpg'%(count_matches)
                        save_name_j = '%.7d_p.jpg'%(count_matches)
                        
                        save_path_i = os.path.join(opts.save_dir, 'regTest', 'data', save_name_i)
                        save_path_j = os.path.join(opts.save_dir, 'regTest', 'data', save_name_j)
                        
                        for data_pairs in zip([load_path_i, load_path_j], [save_path_i, save_path_j], [view_i['alignData'], view_j['alignData']], [img_open_i, img_open_j]):
                            load_path = data_pairs[0]
                            save_path = data_pairs[1]
                            align_data = data_pairs[2]
                            img_curr = data_pairs[3]

                            # Crop a 192x192 image centered around the aligned center
                            l1 = int(float(align_data[1]) - 96)
                            t1 = int(float(align_data[2]) - 96)
                            r1 = l1 + 192 - 1
                            b1 = t1 + 192 - 1
                            box1 = (l1, t1, r1, b1)
                            img_curr = img_curr.crop(box1)
                            # Resize image to 101x101 
                            img_curr = img_curr.resize([101, 101])
                            img_curr.save(save_path)             

                        # Write out the annotation
                        regpairs_positive.write('%s %s %s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %s\n'\
                                              %(save_name_i, save_name_j, '1', rel_pose[0],\
                                              rel_pose[1], rel_pose[2], rel_trans[0],\
                                              rel_trans[1], rel_trans[2], b_angle, str(0)))
                        # A match has been found within this target
                        #found_match_for_target = True
                        count_matches += 1

        if count_targets % 100 == 0:
            sys.stdout.write('Finished target [%d/%d]\n'%(count_targets, len(targets)))
            sys.stdout.flush()
    
    print('\n')
regpairs_positive.close()
