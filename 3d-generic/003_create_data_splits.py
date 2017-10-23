"""
This script creates the training data splits for both the pose
regression and wide baseline stereo matching tasks. 
"""

"""
Import the necessary libraries here
"""
import math
import subprocess
import pdb
import pickle
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import *

dataset_dir = 'dataset/train/'

for base_dir in ['0002/', '0003/', '0004/', '0009/', '0012/', '0014/', '0015/', '0017/', '0020/']:

    targets = create_target_cache(dataset_dir, base_dir) 
    targetIDs = targets.keys()



