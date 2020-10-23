#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:34:20 2020

@author: prithvi
"""

import numpy as np
import io
import cv2
import time
import matplotlib.pyplot as plt
import os
from dataHandler import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import math
from keypoints_Prithvi import *
from m2bk import *
import transformation as tf
from tools import *


dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")

gt = np.loadtxt("japanesealley_sample_P007/P007/pose_left.txt")

#dataset_handler = DatasetHandler("abandonedfactory_sample_P001/P001/")
#gt = np.loadtxt("abandonedfactory_sample_P001/P001/pose_left.txt")

k = dataset_handler.k

num_images = dataset_handler._total_files()
images = []
for i in range(num_images):
    images.append(dataset_handler._read_image_left(i))
    
depth_maps = []
for i in range(num_images):
    depth_maps.append(dataset_handler._read_depth_left(i))
    
kp_list, des_list = extract_features_dataset(images, extract_features)
matches = match_features_dataset(des_list, match_features)

dist_threshold = 0.3

filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)

if len(filtered_matches) > 0:
    
    # Make sure that this variable is set to True if you want to use filtered matches further in your assignment
    is_main_filtered_m = True
    if is_main_filtered_m: 
        matches = filtered_matches

trajectory, orientation = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)



gt1 = shift0(gt)

gt_final = ned2cam(gt1)

gt_x = gt_final[:,0]
gt_y = gt_final[:, 1]
gt_z = gt_final[:, 2]

true = np.stack((gt_x, gt_y, gt_z), axis=0)
visualize_trajectory([np.array(trajectory),np.array(true)])

