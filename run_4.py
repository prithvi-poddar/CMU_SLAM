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
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares, leastsq


dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")

gt = np.loadtxt("japanesealley_sample_P007/P007/pose_left.txt")
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

dist_threshold = 0.2

filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)

if len(filtered_matches) > 0:
    
    # Make sure that this variable is set to True if you want to use filtered matches further in your assignment
    is_main_filtered_m = True
    if is_main_filtered_m: 
        matches = filtered_matches
        
###############################################################################
############################## T E S T ######################################## 
###############################################################################

def fun(camera_params, k, points_2d, points_3d):
    rvec = np.array([camera_params[0],camera_params[1], camera_params[2]]).reshape(3,1)
    tvec = np.array([camera_params[3],camera_params[4], camera_params[5]]).reshape(3,1)
    distCoeffs = np.zeros(4)
    proj,_ = cv2.projectPoints(points_3d, rvec, tvec, k, distCoeffs)
    proj = proj.reshape(len(proj),2)
    return(points_2d.ravel()-proj.ravel())


def estimate_motion_temp(match, kp1, kp2, k, depth1=None):
    
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
    ### START CODE HERE ###
    count=0
    
    
    for m in match:
            
        image1_x, image1_y = kp1[m.queryIdx].pt
        depth = depth1[int(image1_y), int(image1_x)]

        if(depth<900):
            image1_points.append([image1_x, image1_y])

            image1_3D = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]).reshape([3,1]))
            if count==0:
                objectPoints = image1_3D
            else:
                objectPoints = np.c_[objectPoints, image1_3D]

            image2_x, image2_y = kp2[m.trainIdx].pt
            image2_points.append([image2_x, image2_y])
            count+=1

    imagePoints = np.array(image2_points, dtype=np.float64)
    objectPoints = np.transpose(objectPoints)
    distCoeffs = np.zeros(4)
    #print(objectPoints.shape)
        
    _, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, k, distCoeffs)
    new_3d=[]
    new_2d=[]
    for i in inliers:
        new_3d.append(objectPoints[i])
        new_2d.append(imagePoints[i])
        
    new_3d=np.asarray(new_3d).reshape(len(new_3d),3)
    new_2d=np.asarray(new_2d).reshape(len(new_2d),2)
    
    camera_params = np.array([rvec[0][0], rvec[1][0], rvec[2][0], tvec[0][0], tvec[1][0], tvec[2][0]])
    points_3d = new_3d
    points_2d = new_2d
    
    
    x0 = camera_params.ravel()
    f0 = fun(x0, k, points_2d, points_3d)
    
    #res = leastsq(fun, x0, args=(k, points_2d, points_3d))
    res = least_squares(fun, x0, verbose=2,  ftol=1e-15, method='trf', args=(k, points_2d, points_3d))
    #x_scale='jac',
    X = res.x
    rvec = X[:3].reshape(3,1)
    tvec = X[3:].reshape(3,1)
    
    rmat, _ = cv2.Rodrigues(rvec)
    
    return rmat, tvec, image1_points, image2_points


def estimate_trajectory_temp(estimate_motion, matches, kp_list, k, depth_maps=[]):
    
    trajectory = [np.array([0, 0, 0])]
    orientation = [np.array([0, 0, 0, 1])]
    P = np.eye(4)
    
    for i in range(len(matches)):
        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        depth = depth_maps[i]

        rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth)
        R = rmat
        t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
        P_new = np.eye(4)
        P_new[0:3,0:3] = R.T
        P_new[0:3,3] = (-R.T).dot(t)
        P = P.dot(P_new)
        trajectory.append(P[:3,3])
        orientation.append(tf.SO2quat(P[0:3, 0:3]))
        

    trajectory = np.array(trajectory).T
    
    
    return trajectory, orientation

###############################################################################
############################### E N D #########################################
###############################################################################



trajectory, orientation = estimate_trajectory_temp(estimate_motion_temp, matches, kp_list, k, depth_maps=depth_maps)



gt1 = shift0(gt)

gt_final = ned2cam(gt1)

gt_x = gt_final[:,0]
gt_y = gt_final[:, 1]
gt_z = gt_final[:, 2]

true = np.stack((gt_x, gt_y, gt_z), axis=0)
visualize_trajectory([np.array(trajectory),np.array(true)])
