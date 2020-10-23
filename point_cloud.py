#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:30:07 2020

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
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares, leastsq
import open3d as o3d

dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")
gt = np.loadtxt("japanesealley_sample_P007/P007/pose_left.txt")
#dataset_handler = DatasetHandler("abandonedfactory_sample_P001/P001/")
#gt = np.loadtxt("abandonedfactory_sample_P001/P001/pose_left.txt")


k = dataset_handler.k

n = 50
img1 = dataset_handler._read_image_left(n)
depth = dataset_handler._read_depth_left(n)
points_3d = []
colors = []
for x in range(640):
    for y in range(480):
        image1_3D = np.dot(np.linalg.inv(k), np.array([x*depth[y][x], y*depth[y][x], depth[y][x]]))
        image1_color = np.array([img1[y][x][0], img1[y][x][1], img1[y][x][2]])
        points_3d.append(image1_3D)
        colors.append(image1_color)
        
points_3d = np.asarray(points_3d)
colors = np.asarray(colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d[:,:3])
pcd.colors = o3d.utility.Vector3dVector(colors/255)
o3d.visualization.draw_geometries([pcd])
















