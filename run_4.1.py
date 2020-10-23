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

img1 = images[0]
img2 = images[1]
kp1 = kp_list[0]
kp2 = kp_list[1]
depth1 = depth_maps[0]
match = matches[0]


image1_points = []
image2_points = []
count = 0
for m in match:
    image1_x, image1_y = kp1[m.queryIdx].pt
    image1_x = int(image1_x)
    image1_y = int(image1_y)
    depth = depth1[int(image1_y), int(image1_x)]
    
    image1_points.append([image1_x, image1_y])

    image1_3D = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]).reshape([3,1]))
    if count==0:
        objectPoints = image1_3D
    else:
        objectPoints = np.c_[objectPoints, image1_3D]

    image2_x, image2_y = kp2[m.trainIdx].pt
    image2_x = int(image2_x)
    image2_y = int(image2_y)
    image2_points.append([image2_x, image2_y])
    count+=1
    
imagePoints = np.array(image2_points, dtype=np.float64)
objectPoints = np.transpose(objectPoints)
distCoeffs = np.zeros(4)

_, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, k, distCoeffs)
new_3d=[]
new_2d=[]
for i in inliers:
    new_3d.append(objectPoints[i])
    new_2d.append(imagePoints[i])
    
new_3d=np.asarray(new_3d).reshape(len(new_3d),3)
new_2d=np.asarray(new_2d).reshape(len(new_2d),2)


points_3d = new_3d
points_2d = new_2d
    
    
def fun(camera_params, k, points_2d, points_3d):
    rvec = np.array([camera_params[0],camera_params[1], camera_params[2]]).reshape(3,1)
    tvec = np.array([camera_params[3],camera_params[4], camera_params[5]]).reshape(3,1)
    distCoeffs = np.zeros(4)
    proj,_ = cv2.projectPoints(points_3d, rvec, tvec, k, distCoeffs)
    proj = proj.reshape(len(proj),2)
    return(proj.ravel()-points_2d.ravel())**2

#for i in range(10):
camera_params = np.array([rvec[0][0], rvec[1][0], rvec[2][0], tvec[0][0], tvec[1][0], tvec[2][0]])
x0 = camera_params.ravel()
f0 = fun(x0, k, points_2d, points_3d)


res = least_squares(fun, x0, verbose=2, jac='3-point', ftol=1e-15, method='lm', args=(k, points_2d, points_3d))
#x_scale='jac'

X = res.x
rvec = X[:3].reshape(3,1)
tvec = X[3:].reshape(3,1)

###############################################################################
############################### E N D #########################################
###############################################################################

"""
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

        if(depth<15):
            image1_points.append([image1_x, image1_y])

            image1_3D = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]).reshape([3,1]))
            if count==0:
                objectPoints = image1_3D
            else:
                objectPoints = np.c_[objectPoints, image1_3D]

            image2_x, image2_y = kp2[m.trainIdx].pt
            image2_points.append([image2_x, image2_y])
            count+=1

    imagePoints = np.array(image2_points)
    imagePoints = imagePoints.astype('float64')
    objectPoints = np.transpose(objectPoints)
    distCoeffs = np.zeros(4)
    #print(objectPoints.shape)
        
    _, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, k, distCoeffs)
    
    return rvec, tvec, image1_points, image2_points, objectPoints



kp1 = kp_list[0]
kp2 = kp_list[1]
match = matches[0]
depth = depth_maps[0]

rvec, tvec, image1_points, image2_points, objectPoints = estimate_motion_temp(match, kp1, kp2, k, depth)

rmat, _ = cv2.Rodrigues(rvec)

camera_params = np.concatenate((rmat.ravel(), tvec.reshape(3)))


points_3d = objectPoints
points_2d = np.asarray(image2_points)
camera_indices = np.zeros(len(points_2d), dtype=np.int64)
point_indices = np.arange(0, len(points_2d), dtype=np.int64)

n_cameras = 1 #camera_params.shape[0]
n_points = points_3d.shape[0]

n = 12 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))



def project(points, camera_params, k):
    
    
    R = camera_params[:9].reshape(3,3)
    t = camera_params[9:12].reshape(3,1)
    
    P = np.concatenate((R, t), axis=1)
    
    proj = []
    
    for i in range(len(points)):
        projected = k.dot(P).dot(np.array([[points_3d[i][0]], [points_3d[i][1]], [points_3d[i][2]], [1]]))
        proj.append([projected[0][0]/projected[2][0]])
        proj.append([projected[1][0]/projected[2][0]])
        
    return np.asarray(proj)
    


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, k):
    
    camera_params = params[:n_cameras * 12]
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    points_proj = project(points_3d, camera_params, k)
    return (points_proj - points_2d.ravel().reshape(len(points_2d)*2,1)).reshape(len(points_2d)*2)

from scipy.sparse import lil_matrix

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(12):
        A[2 * i, camera_indices * 12 + s] = 1
        A[2 * i + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A


x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, k)
plt.plot(f0)

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

import time
from scipy.optimize import least_squares


t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d, k))
t1 = time.time()

plt.plot(res.fun)


"""