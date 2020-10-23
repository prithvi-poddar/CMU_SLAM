#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 01:37:46 2020

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
#________________________________________________________________________________

###########################################################################################################################
###   For running with downloaded dataset 
###########################################################################################################################

#   Definations__________________________________________________________________

def extract_features(image, binning=True, from_edge=True, num_feat_per_grid=5):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a image
    binning -- True or False for performing feature binning (Default--True)
    from_edge -- True or False for extracting features using edge detection (Default--True)
    num_feat_per_grid -- number of features to retain from each grid if binning=True (Default--5)

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
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
                    
        return new_kp, np.asarray(new_des)
    
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
                    
        return new_kp, np.asarray(new_des)

def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images
    
    """
    kp_list = []
    des_list = []

    for i in images:
        kp, des = extract_features_function(i)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list

def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)

def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck = True)
    # Match descriptors.
    match = bf.match(des1,des2)

    return match

def match_features_dataset(des_list, match_features):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
               
    """
    matches = []

    for i in range(len(des_list)-1):
        matches.append(match_features(des_list[i],des_list[i+1]))
    
    return matches

def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    max_dist = 0

    for m in match:
        if m.distance>max_dist:
            max_dist = m.distance
    for m in match:
        if max_dist != 0:
            if m.distance/max_dist < dist_threshold:
                filtered_match.append(m)

    return filtered_match

def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.imshow(image_matches)
    plt.show()
    plt.pause(0.01)


def dynamic_masking(seg_image, kp, des, masking_values):
    """
    Remove keypoints of dynamic objects

    Arguments:
    seg_image -- the segmented image of the scene
    kp -- list of the keypoints
    des -- list of the descriptions
    masking_values -- [Nx1] array of the pixel values of the dynamic objects in the segmented image

    Returns:
    new_kp -- list of feature points with the dynamic objects removed
    new_des -- descriptions of the new feature points 
    """
    
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
