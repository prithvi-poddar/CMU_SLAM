#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 01:01:17 2020

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
from keypoints import *

"""
def extract_features(image, binning=True, from_edge=True, num_feat_per_grid=5):
    if (binning==False and from_edge==False):
        orb = cv2.ORB_create(nfeatures=5000)
        kp, des = orb.detectAndCompute(image,None)
        return kp, des
    
    elif(binning==False and from_edge==True):
        orb = cv2.ORB_create(nfeatures=5000)
        edges = cv2.Canny(image,50, 100, L2gradient=True)
        kp_edge = orb.detect(edges,None)
        kp_edge, des_edge = orb.compute(image, kp_edge)
        return kp_edge, des_edge
    
    elif(binning==True and from_edge==False):
        orb = cv2.ORB_create(nfeatures=5000)
        kp, des = orb.detectAndCompute(image,None)
        ####GRIDDING####
        h = np.arange(0, 480, 40).astype(np.float64)
        w = np.arange(0, 640, 40).astype(np.float64)
        keys=[]
        for i in h:
            for j in w:
                temporary = []
                for a in range(len(kp)):
                    if (kp[a].pt[0]>=j and kp[a].pt[0]<=j+39.0 and kp[a].pt[1]>=i and kp[a].pt[1]<=i+39.0):
                        temp = [kp[a], des[a]]
                        temporary.append(temp)
                        
                if (len(temporary)>0):
                    keys.append(temporary)
        ####SORTING###                
        for a in range(len(keys)):
            num = len(keys[a])
            
            for i in range(num):
                for j in range(0, num-i-1):
                    if keys[a][j][0].response < keys[a][j+1][0].response:
                        te = keys[a][j]
                        keys[a][j] = keys[a][j+1]
                        keys[a][j+1] = te
        ####FILTERING####
        new_kp=[]
        new_des=[]
        to_keep = num_feat_per_grid
        for i in range(len(keys)):
            if len(keys[i])<to_keep:
                for j in range(len(keys[i])):
                    new_kp.append(keys[i][j][0])
                    new_des.append(keys[i][j][1])
                    
            elif len(keys[i])>=to_keep:
                for j in range(to_keep):
                    new_kp.append(keys[i][j][0])
                    new_des.append(keys[i][j][1])
                    
        return new_kp, new_des
    
    elif (binning==True and from_edge==True):
        orb = cv2.ORB_create(nfeatures=5000)
        edges = cv2.Canny(image,50, 100, L2gradient=True)
        kp_edge = orb.detect(edges,None)
        kp_edge, des_edge = orb.compute(image, kp_edge)
        ####GRIDDING####
        h = np.arange(0, 480, 40).astype(np.float64)
        w = np.arange(0, 640, 40).astype(np.float64)
        keys=[]
        
        for i in h:
            for j in w:
                temporary = []
                for a in range(len(kp_edge)):
                    if (kp_edge[a].pt[0]>=j and kp_edge[a].pt[0]<=j+39.0 and kp_edge[a].pt[1]>=i and kp_edge[a].pt[1]<=i+39.0):
                        temp = [kp_edge[a], des_edge[a]]
                        temporary.append(temp)
                        
                if (len(temporary)>0):
                    keys.append(temporary)
        ####SORTING####
        for a in range(len(keys)):
            num = len(keys[a])
            
            for i in range(num):
                for j in range(0, num-i-1):
                    if keys[a][j][0].response < keys[a][j+1][0].response:
                        te = keys[a][j]
                        keys[a][j] = keys[a][j+1]
                        keys[a][j+1] = te
        ####FILTERING####
        new_kp=[]
        new_des=[]
        to_keep = num_feat_per_grid
        for i in range(len(keys)):
            if len(keys[i])<to_keep:
                for j in range(len(keys[i])):
                    new_kp.append(keys[i][j][0])
                    new_des.append(keys[i][j][1])
                    
            elif len(keys[i])>=to_keep:
                for j in range(to_keep):
                    new_kp.append(keys[i][j][0])
                    new_des.append(keys[i][j][1])
                    
        return new_kp, new_des
        
"""

def dynamic_masking(seg_image, kp, des, masking_values):
    idx = []
    for mask in masking_values:
        for i in range(len(kp)):
            if (seg_image[int(kp[i].pt[1])][int(kp[i].pt[0])] == mask):
                idx.append(i)
                
    new_kp = []
    new_des = []
    for i in range(len(kp)):
        if (i in idx):
            pass
        else:
            new_kp.append(kp[i])
            new_des.append(des[i])
    return new_kp, np.asarray(new_des)


        
        
dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")
n = 72
left = dataset_handler._read_image_left(n)
right = dataset_handler._read_image_right(n)
left_seg = dataset_handler._read_seg_left(n)


kp_left, des_left = extract_features(left, binning=True, from_edge=False, num_feat_per_grid=30)
kp_right, des_right = extract_features(right, binning=True, from_edge=False, num_feat_per_grid=30)

kp_left, des_left = dynamic_masking(left_seg, kp_left, des_left, [244])

visualize_features(left, kp_left)

"""
match = match_features(des_left, des_right)

dist_threshold = 0.4
match = filter_matches_distance(match, dist_threshold)

image_matches = visualize_matches(left, kp_left, right, kp_right, match) 

"""





