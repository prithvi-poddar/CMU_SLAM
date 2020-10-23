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
from keypoints_Prithvi import *


dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")

left1 = dataset_handler._read_image_left(0)
left2 = dataset_handler._read_image_left(1)

K = dataset_handler.k



sift = cv2.xfeatures2d.SIFT_create()



kp1, des1 = sift.detectAndCompute(cv2.cvtColor(left1, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = sift.detectAndCompute(cv2.cvtColor(left2, cv2.COLOR_BGR2GRAY), None)



#img=cv2.drawKeypoints(left1_g,kp1,left1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.imshow(img)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1,des2)

#image_matches = cv2.drawMatches(left1,kp1,left2,kp2,matches,None)
#plt.imshow(image_matches)

####################################################################################

left1_pt = []
left2_pt = []
for m in matches:
    left1_pt.append(kp1[m.queryIdx].pt)
    left2_pt.append(kp2[m.trainIdx].pt)

left1_pt = np.asarray(left1_pt)
left2_pt = np.asarray(left2_pt)
F, mask = cv2.findFundamentalMat(left1_pt, left2_pt, cv2.FM_RANSAC)


left1_pt = left1_pt[mask.ravel()==1]
left2_pt = left2_pt[mask.ravel()==1]

def skew_symmetric(x):
    res = [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]
    return np.asarray(res)

E = ((K.T).dot(F)).dot(K)

R1, R2, t =	cv2.decomposeEssentialMat(E)

P1 = np.concatenate((np.eye(3), [[0],[0],[0]]), axis=1)
P21 = np.concatenate((R1, t), axis=1)
P22 = np.concatenate((R2, t), axis=1)
P23 = np.concatenate((R1, -t), axis=1)
P24 = np.concatenate((R2, -t), axis=1)

##### POINT TRIANGULATION ####

left1_image_homo = np.concatenate((left1_pt, np.ones(len(left1_pt), dtype=np.float64).reshape(len(left1_pt), 1)), axis=1)
left2_image_homo = np.concatenate((left2_pt, np.ones(len(left1_pt), dtype=np.float64).reshape(len(left1_pt), 1)), axis=1)


left1_image_skew = []
left2_image_skew = []

for i in range(len(left1_image_homo)):
    left1_image_skew.append(skew_symmetric(left1_image_homo[i]))
    left2_image_skew.append(skew_symmetric(left2_image_homo[i]))





def triangulate(left1_skew, P1, left2_skew, P2):
    A1 = left1_skew.dot(P1)
    A2 = left2_skew.dot(P2)
    A = np.concatenate((A1, A2), axis = 0)
    _, __, vh = np.linalg.svd(A)
    v = vh.T
    X = v[:, 2]
    X_world = np.asarray([[X[0]/X[3]], [X[1]/X[3]], [X[2]/X[3]]])
    return X_world




X_world1 = triangulate(left1_image_skew[2], P1, left2_image_skew[2], P21)
X_world2 = triangulate(left1_image_skew[3], P1, left2_image_skew[3], P21)

X_world = []
for i in range(len(left1_image_skew)):
    X_world.append(triangulate(left1_image_skew[i], P1, left2_image_skew[i], P21))



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
    
    

R, T = disambiguate_transform(R1, R2, t, left1_pt, left2_pt)

#### FINDING SCALE OF TRANSLATION ####

left1_depth_map = dataset_handler._read_depth_left(0)
left2_depth_map = dataset_handler._read_depth_left(1)

left1_depth = left1_depth_map[int(left1_pt[0][1])][int(left1_pt[0][0])]
left2_depth = left2_depth_map[int(left2_pt[0][1])][int(left2_pt[0][0])]

left1_3D = np.dot(np.linalg.inv(K), np.array([int(left1_pt[0][0])*left1_depth, int(left1_pt[0][1])*left1_depth, left1_depth]).reshape([3,1]))
left2_3D = np.dot(np.linalg.inv(K), np.array([int(left2_pt[0][0])*left2_depth, int(left2_pt[0][1])*left2_depth, left2_depth]).reshape([3,1]))

left1_img_3D = triangulate(left1_image_skew[0], P1, left2_image_skew[0], P21)


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
        
        r = math.sqrt(x1[0]**2 + x1[1]**2 + x1[2]**2)/(math.sqrt(x2[0]**2 + x2[1]**2 + x2[2]**2)+0.01)
        
        #r = np.linalg.norm(X_3D[a] - X_3D[a+1])/np.linalg.norm(X_world[a] - X_world[a+1])
        scale = scale + r
        a+=2
        count+=1
    
    return X_3D, X_world



X_3D, X_world = find_scale(left1_depth_map, left1_pt, left2_pt, R, T)
        
