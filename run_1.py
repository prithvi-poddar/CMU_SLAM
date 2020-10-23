#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:41:56 2020

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
#from keypoints_Prithvi import *




def skew_symmetric(x):
    res = [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]
    return np.asarray(res)

def triangulate(left1_skew, P1, left2_skew, P2):
    A1 = left1_skew.dot(P1)
    A2 = left2_skew.dot(P2)
    A = np.concatenate((A1, A2), axis = 0)
    _, __, vh = np.linalg.svd(A)
    v = vh.T
    X = v[:, 2]
    X_world = np.asarray([[X[0]/X[3]], [X[1]/X[3]], [X[2]/X[3]]])
    return X_world

def disambiguate_transform(R1, R2, t, left1_pt, left2_pt):
    P1 = np.concatenate((np.eye(3), [[0],[0],[0]]), axis=1)
    P21 = np.concatenate((R1, t), axis=1)
    P22 = np.concatenate((R2, t), axis=1)
    P23 = np.concatenate((R1, -t), axis=1)
    P24 = np.concatenate((R2, -t), axis=1)
    
    left1_image_homo = np.concatenate((left1_pt, np.ones(len(left1_pt), dtype=np.float64).reshape(len(left1_pt), 1)), axis=1)
    left2_image_homo = np.concatenate((left2_pt, np.ones(len(left1_pt), dtype=np.float64).reshape(len(left1_pt), 1)), axis=1)
    
    left1_image_skew = []
    left2_image_skew = []
    
    for i in range(len(left1_image_homo)):
        left1_image_skew.append(skew_symmetric(left1_image_homo[i]))
        left2_image_skew.append(skew_symmetric(left2_image_homo[i]))
        
        
    #### TESTING CONF 1 ####
    X_world = []
    for i in range(len(left1_image_skew)):
        X_world.append(triangulate(left1_image_skew[i], P1, left2_image_skew[i], P21))
        
    r3_1 = np.array([0, 0, 1], dtype=np.float64)
    c1 = P1[:,3].reshape(3,1)
    r3_2 = R1[2,:]
    c2 = t
    
    conf1_total = 0
    for i in range(len(X_world)):
        if (r3_1.dot(X_world[i]-c1)[0] > 0 and r3_2.dot(X_world[i]-c2)[0] > 0):
            conf1_total += 1
     

    #### TESTING CONF 2 ####
    X_world = []
    for i in range(len(left1_image_skew)):
        X_world.append(triangulate(left1_image_skew[i], P1, left2_image_skew[i], P22))
        
    r3_1 = np.array([0, 0, 1], dtype=np.float64)
    c1 = P1[:,3].reshape(3,1)
    r3_2 = R2[2,:]
    c2 = t
    
    conf2_total = 0
    for i in range(len(X_world)):
        if (r3_1.dot(X_world[i]-c1)[0] > 0 and r3_2.dot(X_world[i]-c2)[0] > 0):
            conf2_total += 1
            
            
    #### TESTING CONF 3 ####
    X_world = []
    for i in range(len(left1_image_skew)):
        X_world.append(triangulate(left1_image_skew[i], P1, left2_image_skew[i], P23))
        
    r3_1 = np.array([0, 0, 1], dtype=np.float64)
    c1 = P1[:,3].reshape(3,1)
    r3_2 = R1[2,:]
    c2 = -t
    
    conf3_total = 0
    for i in range(len(X_world)):
        if (r3_1.dot(X_world[i]-c1)[0] > 0 and r3_2.dot(X_world[i]-c2)[0] > 0):
            conf3_total += 1
            
    
    
    #### TESTING CONF 4 ####
    X_world = []
    for i in range(len(left1_image_skew)):
        X_world.append(triangulate(left1_image_skew[i], P1, left2_image_skew[i], P24))
        
    r3_1 = np.array([0, 0, 1], dtype=np.float64)
    c1 = P1[:,3].reshape(3,1)
    r3_2 = R2[2,:]
    c2 = -t
    
    conf4_total = 0
    for i in range(len(X_world)):
        if (r3_1.dot(X_world[i]-c1)[0] > 0 and r3_2.dot(X_world[i]-c2)[0] > 0):
            conf4_total += 1
    
    
    #### FINDING THE CORRECT CONF ####
    
    results = np.array([conf1_total, conf2_total, conf3_total, conf4_total])
    final = np.argmax(results)
    
    if (final == 0):
        return R1, t
    elif (final == 1):
        return R2, t
    elif (final == 2):
        return R1, -t
    else:
        return R2, -t
    
def find_scale(left1_depth_map, left1_pt, left2_pt, R, T):
    P1 = np.concatenate((np.eye(3), [[0],[0],[0]]), axis=1)
    P2 = np.concatenate((R, T), axis=1)
    
    left1_image_homo = np.concatenate((left1_pt, np.ones(len(left1_pt), dtype=np.float64).reshape(len(left1_pt), 1)), axis=1)
    left2_image_homo = np.concatenate((left2_pt, np.ones(len(left2_pt), dtype=np.float64).reshape(len(left2_pt), 1)), axis=1)
    
    left1_image_skew = []
    left2_image_skew = []
    
    for i in range(len(left1_image_homo)):
        left1_image_skew.append(skew_symmetric(left1_image_homo[i]))
        left2_image_skew.append(skew_symmetric(left2_image_homo[i]))
        
        
    #### TESTING CONF 1 ####
    X_world = []
    for i in range(len(left1_image_skew)):
        X_world.append(triangulate(left1_image_skew[i], P1, left2_image_skew[i], P2))
        
    X_3D = []
    for i in range(len(left1_pt)):
        left1_depth = left1_depth_map[int(left1_pt[i][1])][int(left1_pt[i][0])]
        left1_3D = np.dot(np.linalg.inv(K), np.array([int(left1_pt[i][0])*left1_depth, int(left1_pt[i][1])*left1_depth, left1_depth]).reshape([3,1]))
        X_3D.append(left1_3D)
        
    a = 0
    count = 0
    scale = 0
    while (a+2 < len(X_world)):
        x1 = X_3D[a] - X_3D[a+1]
        x2 = X_world[a] - X_world[a+1]
        
        if (np.absolute(math.sqrt(x1[0][0]**2 + x1[1][0]**2 + x1[2][0]**2) > 0)):
            r = (math.sqrt(x2[0][0]**2 + x2[1][0]**2 + x2[2][0]**2))/math.sqrt(x1[0][0]**2 + x1[1][0]**2 + x1[2][0]**2)
            #r = np.linalg.norm(X_3D[a] - X_3D[a+1])/np.linalg.norm(X_world[a] - X_world[a+1])
            scale = scale + r
            count+=1
            
        a += 2
    
    return scale/count

def extract_features(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    
    return kp, des

def extract_features_dataset(images, extract_features_function):
    
    kp_list = []
    des_list = []

    for i in images:
        kp, des = extract_features_function(i)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list

def match_features(des1, des2):
 
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)

    return matches

def match_features_dataset(des_list, match_features):
    matches = []
    for i in range(len(des_list)-1):
        matches.append(match_features(des_list[i],des_list[i+1]))

    return matches

def estimate_motion(image1, image2, depth_map, K):
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
        
    img1_pt = []
    img2_pt = []
    for m in matches:
        img1_pt.append(kp1[m.queryIdx].pt)
        img2_pt.append(kp2[m.trainIdx].pt)
        
    img1_pt = np.asarray(img1_pt)
    img2_pt = np.asarray(img2_pt)
    F, mask = cv2.findFundamentalMat(img1_pt, img2_pt, cv2.FM_RANSAC)
    
    img1_pt = img1_pt[mask.ravel()==1]
    img2_pt = img2_pt[mask.ravel()==1]
    
    E = ((K.T).dot(F)).dot(K)
    R1, R2, t =	cv2.decomposeEssentialMat(E)
    
    R, T = disambiguate_transform(R1, R2, t, img1_pt, img2_pt)
    
    scale = find_scale(depth_map, img1_pt, img2_pt, R, T)
    
    return R, -T*scale

def estimate_trajectory(estimate_motion, images, depth_maps, K):
    
    trajectory = [np.array([0, 0, 0])]
    P = np.eye(4)
    
    for i in range(len(images)-1):
        
        depth = depth_maps[i]

        rmat, tvec = estimate_motion(images[i], images[i+1], depth_maps[i], K)
        R = rmat
        t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
        P_new = np.eye(4)
        P_new[0:3,0:3] = R.T
        P_new[0:3,3] = (-R.T).dot(t)
        P = P.dot(P_new)
        trajectory.append(P[:3,3])
        

    trajectory = np.array(trajectory).T
    
    
    return trajectory

def estimate_trajectory_temp(estimate_motion, image1, image2, depth, P, K):
    rmat, tvec = estimate_motion(image1, image2, depth, K )
    
    R = rmat
    t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
    P_new = np.eye(4)
    P_new[0:3,0:3] = R.T
    P_new[0:3,3] = (-R.T).dot(t)
    P = P.dot(P_new)
    trajectory = P[:3,3]  
    return trajectory, P



def visualize_trajectory(rt):
    tr = []
    for trajectory in rt:
        # Unpack X Y Z each trajectory point
        locX = []
        locY = []
        locZ = []

        max = -math.inf
        min = math.inf

        # Needed for better visualisation
        maxY = -math.inf
        minY = math.inf

        for i in range(0, trajectory.shape[1]):
            current_pos = trajectory[:, i]
            
            locX.append(current_pos.item(0))
            locY.append(current_pos.item(1))
            locZ.append(current_pos.item(2))
            if np.amax(current_pos) > max:
                max = np.amax(current_pos)
            if np.amin(current_pos) < min:
                min = np.amin(current_pos)

            if current_pos.item(1) > maxY:
                maxY = current_pos.item(1)
            if current_pos.item(1) < minY:
                minY = current_pos.item(1)

        auxY_line = locY[0] + locY[-1]
        if max > 0 and min > 0:
            minY = auxY_line - (max - min) / 2
            maxY = auxY_line + (max - min) / 2
        elif max < 0 and min < 0:
            minY = auxY_line + (min - max) / 2
            maxY = auxY_line - (min - max) / 2
        else:
            minY = auxY_line - (max - min) / 2
            maxY = auxY_line + (max - min) / 2
        tr.append([locX,locY,locZ])

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    gspec = gridspec.GridSpec(2, 2)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])

    D3_plt = plt.subplot(gspec[0, 0], projection='3d')
    
    tr = np.array(tr)

    for u in range(len(rt)):   
        if u==0:
            D3_plt.plot3D(tr[u,0], tr[u,1], tr[u,2], zorder=0,color = 'green')
            traj_main_plt.plot( tr[u,2], tr[u,0], ".-", label="Trajectory", zorder=1, linewidth=2, markersize=2)
            YX_plt.plot(tr[u,1], tr[u,0], ".-", linewidth=1, markersize=4, zorder=0)
            # YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
            # YX_plt.scatter([0], [0], s=2, c="red", label="Start location", zorder=2)
            ZY_plt.plot(tr[u,2], tr[u,1], ".-", linewidth=1, markersize=4, zorder=0)
            # ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
            # ZY_plt.scatter([0], [0], s=2, c="red", label="Start location", zorder=2)
        elif u==1:
            D3_plt.plot3D(tr[u,0], tr[u,1], tr[u,2], zorder=0,color = 'purple')
            traj_main_plt.plot( tr[u,2], tr[u,0], ".-", label="Trajectory gt", zorder=1, linewidth=1, markersize=2)
            YX_plt.plot(tr[u,1], tr[u,0], ".-", linewidth=1, markersize=4, zorder=0)
            # YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
            # YX_plt.scatter([0], [0], s=2, c="red", label="Start location", zorder=2)
            ZY_plt.plot(tr[u,2], tr[u,1], ".-", linewidth=1, markersize=4, zorder=0)
            # ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
            # ZY_plt.scatter([0], [0], s=2, c="red", label="Start location", zorder=2)
        else:
            D3_plt.scatter(tr[u,0], tr[u,1], tr[u,2],s = 1 ,zorder=0) 

    # Actual trajectory plotting ZX
    
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.set_xlabel("Z")
    traj_main_plt.axes.yaxis.set_ticklabels([])
    # Plot reference lines
    traj_main_plt.plot([locZ[0], locZ[-1]], [locX[0], locX[-1]], "--", label="Auxiliary line", zorder=0, linewidth=1)
    # Plot camera initial location
    traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    # traj_main_plt.set_xlim([min, max])
    # traj_main_plt.set_ylim([min, max])
    traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    ZY_plt.set_title("Z Y", y=toffset)
    ZY_plt.set_ylabel("Y", labelpad=-4)
    ZY_plt.axes.xaxis.set_ticklabels([])
    # ZY_plt.set_xlim([min, max])
    # ZY_plt.set_ylim([minY, maxY])

    # Plot YX
    YX_plt.set_title("Y X", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
    # YX_plt.set_xlim([minY, maxY])
    # YX_plt.set_ylim([min, max])
    

    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)
    D3_plt.view_init(90, azim=90)
   


#________________________________________________________________________________

#   Main function________________________________________________________________

dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")
tr_r_gt = dataset_handler._read_pose_right()
tr_l_gt = dataset_handler._read_pose_left()
plt.ion()
fig = plt.figure(figsize=(16, 6), dpi=100)
pt_cld = []
trajectory_l =[tr_l_gt[0,:3]]
projection = [np.eye(4)]
projection[0][0:3,3] = tr_l_gt[0,:3]

traj_r_gt = []
traj_l_gt = []


K = dataset_handler.k
num_images = dataset_handler._total_files()

for i in range(num_images - 1):
    
    image1 = dataset_handler._read_image_left(i)
    image2 = dataset_handler._read_image_left(i+1)
    depth = dataset_handler._read_depth_left(i)
    K = dataset_handler.k
    

    traj_l_gt.append(tr_l_gt[i,:3])
    
    
    traj_l,p = estimate_trajectory_temp(estimate_motion, image1, image2, depth, projection[i], K)
    projection.append(p)
    trajectory_l.append(traj_l)

    visualize_trajectory([np.array(trajectory_l).T,np.array(traj_l_gt).T])#,np.array(pt_cld).T])
    
    plt.show()
    plt.pause(0.01)
    print(i)







"""
images = []
for i in range(num_images):
    images.append(dataset_handler._read_image_left(i))
    
depth_maps = []
for i in range(num_images):
    depth_maps.append(dataset_handler._read_depth_left(i))


trajectory = estimate_trajectory(estimate_motion, images, depth_maps, K)

"""
