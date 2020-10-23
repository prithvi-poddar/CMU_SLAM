#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:02:22 2020

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


""" Getting translation and rotation between image0 and image1"""

kp1 = kp_list[0]
kp2 = kp_list[1]
des1 = des_list[0]
des2 = des_list[1]
depth1 = depth_maps[0]
depth2 = depth_maps[1]
match = matches[0]

r, tvec, _, __ = estimate_motion(match, kp1, kp2, k, depth1)

image1_pt = []
image2_pt = []
for m in match:
    image1_pt.append(kp1[m.queryIdx].pt)
    image2_pt.append(kp2[m.trainIdx].pt)

image1_pt = np.asarray(image1_pt)
image2_pt = np.asarray(image2_pt)

image1_3d = []

for i in range(len(image1_pt)):
    image1_x = int(image1_pt[i][0])
    image1_y = int(image1_pt[i][1])
    depth = depth1[image1_y, image1_x]
    pt_3d = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]).reshape([3,1]))
    image1_3d.append(pt_3d)
    
""" Bundle Adjustment """

R = r.T
t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
C = ((-r.T).dot(t)).reshape([3,1])

""" Generating f(x) and b """

test1 = k.dot(R).dot(image1_3d[0]-C)
test1_ = np.array([[test1[0][0]/test1[2][0]], [test1[1][0]/test1[2][0]]]) 
test1 = test1.reshape((3))

test2 = k.dot(R).dot(image1_3d[1]-C)
test2_ = np.array([[test2[0][0]/test2[2][0]], [test2[1][0]/test2[2][0]]])
test2 = test2.reshape((3))

f_x = np.concatenate((test1_, test2_))
f_x_w = []
f_x_w.append(test1)
f_x_w.append(test2)

for i in range(2, len(image1_3d)):
    temp = k.dot(R).dot(image1_3d[i]-C)
    temp_ = np.array([[temp[0][0]/temp[2][0]], [temp[1][0]/temp[2][0]]])
    f_x = np.concatenate((f_x, temp_))
    temp = temp.reshape((3))
    f_x_w.append(temp)
    
    
b1 = np.array([[image2_pt[0][0]], [image2_pt[0][1]]])
b2 = np.array([[image2_pt[1][0]], [image2_pt[1][1]]])
b = np.concatenate((b1, b2))

for i in range(2, len(image2_pt)):
    temp = np.array([[image2_pt[i][0]], [image2_pt[i][1]]])
    b = np.concatenate((b, temp))





""" generating the jacobian """

def get_jacobian(f_x_w, R, k, C, image1_3d):
    #j1 = np.zeros(2*(len(f_x_w), 7))
    #j2 = np.zeros(2*(len(f_x_w), len(f_x_w)*3))
    f = k[0][0]
    px = k[0][2]
    py = k[1][2]
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]
    
    """ Making j1"""
    
    ##df/dc##
    df_dc_comp = []
    for i in range(len(f_x_w)):
        u = f_x_w[i][0]
        v = f_x_w[i][1]
        w = f_x_w[i][2]
        
        du_dc = -np.array([f*r11 + px*r31, f*r12 + px*r32, f*r13 + px*r33]).reshape((3))
        dv_dc = -np.array([f*r21 + py*r31, f*r22 + py*r32, f*r23 + py*r33]).reshape((3))
        dw_dc = -np.array([r31, r32, r33]).reshape((3))
        
        df_dc1 = ((w*du_dc) - (u*dw_dc))/w**2
        df_dc2 = ((w*dv_dc) - (v*dw_dc))/w**2
        df_dc = np.stack((df_dc1, df_dc2))
        df_dc_comp.append(df_dc)
        
    df_dc_final = np.concatenate((df_dc_comp[0], df_dc_comp[1]))
    for i in range(2, len(df_dc_comp)):
        df_dc_final = np.concatenate((df_dc_final, df_dc_comp[i]))
    
    ##df_dr##
    C1 = C[0][0]
    C2 = C[1][0]
    C3 = C[2][0]
    df_dr_comp = []
    for i in range(len(image1_3d)):
        X1 = image1_3d[i][0][0]
        X2 = image1_3d[i][1][0]
        X3 = image1_3d[i][2][0]
        
        du_dr = np.array([f*(X1-C1), f*(X2-C2), f*(X3-C3), 0, 0, 0, px*(X1-C1), px*(X2-C2), px*(X3-C3)])
        dv_dr = np.array([0, 0, 0, f*(X1-C1), f*(X2-C2), f*(X3-C3), py*(X1-C1), py*(X2-C2), py*(X3-C3)])
        dw_dr = np.array([0, 0, 0, 0, 0, 0, (X1-C1), (X2-C2), (X3-C3)])
        
        df_dr1 = ((w*du_dr) - (u*dw_dr))/w**2
        df_dr2 = ((w*dv_dr) - (v*dw_dr))/w**2
        df_dr = np.stack((df_dr1, df_dr2))
        df_dr_comp.append(df_dr)
    
    df_dr_final = np.concatenate((df_dr_comp[0], df_dr_comp[1]))
    for i in range(2, len(df_dr_comp)):
        df_dr_final = np.concatenate((df_dr_final, df_dr_comp[i]))
        
        
    ##dr_dq##
    q = get_quaternions(R)
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    
    dr_dq_final = np.array([[0, -4*qy, -4*qz, 0],
                            [2*qy, 2*qx, -2*qw, -2*qz],
                            [2*qz, 2*qw, 2*qx, 2*qy],
                            [2*qy, 2*qx, 2*qw, 2*qz],
                            [-4*qx, 0, -4*qz, 0],
                            [-2*qw, 2*qz, 2*qy, 2*qx],
                            [2*qz, -2*qw, 2*qx, -2*qy],
                            [2*qw, 2*qz, 2*qy, 2*qx],
                            [-4*qx, -4*qy, 0, 0]])
    ##df_dq##
    df_dq = df_dr_final.dot(dr_dq_final)
    
    j1 = np.concatenate((df_dq, df_dc_final), axis=1)
    
    
    """ Making j2 """
    
    ### df_dx ###
    
    du_dx = np.array([f*r11 + px*r31, f*r12 + px*r32, f*r13 + px*r33]).reshape((3))
    dv_dx = np.array([f*r21 + py*r31, f*r22 + py*r32, f*r23 + py*r33]).reshape((3))
    dw_dx = np.array([r31, r32, r33]).reshape((3))
    
    df_dx1 = ((w*du_dx) - (u*dw_dx))/w**2
    df_dx2 = ((w*dv_dx) - (v*dw_dx))/w**2
    df_dx = np.stack((df_dx1, df_dx2))
    
    j2_comp = []
    for i in range(len(image1_3d)):
        temp = np.zeros((2*len(image1_3d), 3))
        temp[2*i:(2*i)+2,:] = df_dx
        j2_comp.append(temp)
        
    j2 = np.concatenate((j2_comp[0], j2_comp[1]), axis = 1)
    for i in range(2, len(j2_comp)):
        j2 = np.concatenate((j2, j2_comp[i]), axis=1)
    
    j = np.concatenate((j1, j2), axis=1)
    return j
    
        

j = get_jacobian(f_x_w, R, k, C, image1_3d)

del_x = np.linalg.pinv(j).dot(b-f_x)

dqw = del_x[0][0]
dqx = del_x[1][0]
dqy = del_x[2][0]
dqz = del_x[3][0]
q = get_quaternions(R)

dq = [dqw, dqx, dqy, dqz]

q_new = quatHProd(q, dq)

R = get_rotation_from_quaternion(q_new[0], q_new[1], q_new[2], q_new[3])

dc1 = del_x[4]
dc2 = del_x[5]
dc3 = del_x[6]
dc = np.stack((dc1, dc2, dc3))

C = C+dc


test1 = k.dot(R).dot(image1_3d[0]-C)
test1_ = np.array([[test1[0][0]/test1[2][0]], [test1[1][0]/test1[2][0]]]) 
test1 = test1.reshape((3))

test2 = k.dot(R).dot(image1_3d[1]-C)
test2_ = np.array([[test2[0][0]/test2[2][0]], [test2[1][0]/test2[2][0]]])
test2 = test2.reshape((3))

f_x = np.concatenate((test1_, test2_))
f_x_w = []
f_x_w.append(test1)
f_x_w.append(test2)

for i in range(2, len(image1_3d)):
    temp = k.dot(R).dot(image1_3d[i]-C)
    temp_ = np.array([[temp[0][0]/temp[2][0]], [temp[1][0]/temp[2][0]]])
    f_x = np.concatenate((f_x, temp_))
    temp = temp.reshape((3))
    f_x_w.append(temp)
    
    
b1 = np.array([[image2_pt[0][0]], [image2_pt[0][1]]])
b2 = np.array([[image2_pt[1][0]], [image2_pt[1][1]]])
b = np.concatenate((b1, b2))

for i in range(2, len(image2_pt)):
    temp = np.array([[image2_pt[i][0]], [image2_pt[i][1]]])
    b = np.concatenate((b, temp))

j = get_jacobian(f_x_w, R, k, C, image1_3d)

del_x = np.linalg.pinv(j).dot(b-f_x)