#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:21:18 2020

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