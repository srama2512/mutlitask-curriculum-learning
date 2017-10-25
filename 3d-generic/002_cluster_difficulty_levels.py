"""
This script is used to read the train data, create intermediate target 
caches and also computed some cluster statistics for pre-defined
difficulty levels.
"""

"""
Import the necessary libraries here
"""
import math
import pdb
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import *

"""
Group the matches into categories of 
 (0) beta < 30
 (1) beta < 60
 (2) beta < 90
 (3) beta < 120
 (4) all other beta 
"""

def category_assignment(beta):
    category = 0
    if beta < 30:
        category = 0
    elif beta < 60:
        category = 1
    elif beta < 90:
        category = 2
    elif beta < 120:
        category = 3
    else:
        category = 4
    return category    

"""
Define the base directory and read in the image
and text files
"""
dataset_dir = 'dataset/train/'

for base_dir in ['0017/', '0020/']:

    targets = create_target_cache(dataset_dir, base_dir) 
    targetIDs = targets.keys()

    print("Number of targets: %d"%(len(targetIDs)))
    """
    Divide the matching views based on baseline angle
    """
    categories = [[], [], [], [], []]

    for i in range(len(targetIDs)):
        targetIDCurr = targetIDs[i]
        targetCurr = targets[targetIDCurr]
        #print('\n================================================================\n')
        #print('Number of views for target %d: %d'%(targetIDCurr, len(targetCurr['views'])))
        for j in range(len(targetCurr['views'])-1):
            for k in range(j+1, len(targetCurr['views'])):
                b_angle = baseline_angle_1(targetCurr['views'][j]['cameraCoord'], targetCurr['views'][k]['cameraCoord'], targetCurr['targetCoord'])
                align_data_j = targetCurr['views'][j]['alignData']
                align_data_k = targetCurr['views'][k]['alignData']
                if float(align_data_j[32]) >= 0.15 and float(align_data_k[32]) >= 0.15:
                #print('B angle: %.4f'%(b_angle))
                    #print('Img 1: %s ::: Img 2: %s'%(targetCurr['views'][j]['imagePath'], targetCurr['views'][k]['imagePath']))
                    categories[category_assignment(b_angle)].append([targetCurr['views'][j]['imagePath'], targetCurr['views'][k]['imagePath'], b_angle, align_data_j, align_data_k])

    for i, category in enumerate(categories):
        print('Number of points in category # %d: %d'%(i, len(category)))
