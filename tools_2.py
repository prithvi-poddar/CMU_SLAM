#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:42:21 2020

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
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares

def extract_features(image):
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    
    return kp, des

def visualize_features(image, kp):
    display = cv2.drawKeypoints(image, kp, None)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)
    
def extract_features_dataset(images, extract_features_function):
   
    kp_list = []
    des_list = []
    
    ### START CODE HERE ###
    for i in images:
        kp, des = extract_features_function(i)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list

def match_features(des1, des2):
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    
    return matches

def filter_matches_distance(match, dist_threshold):
    
    filtered_match = []
    max_dist=0
    
    for m in match:
        if m.distance>max_dist:
            max_dist = m.distance
            
    for m in match:
        if m.distance/max_dist < dist_threshold:
            filtered_match.append(m)

    return filtered_match

def filter_matches_distance_by_fundamental(match, kp1, kp2):
    left1_pt = []
    left2_pt = []
    for m in match:
        left1_pt.append(kp1[m.queryIdx].pt)
        left2_pt.append(kp2[m.trainIdx].pt)
    
    left1_pt = np.asarray(left1_pt)
    left2_pt = np.asarray(left2_pt)
    F, mask = cv2.findFundamentalMat(left1_pt, left2_pt, cv2.FM_RANSAC)
    filtered_match = []
    for i in range(len(mask)):
        if (mask[i]==1):
            filtered_match.append(match[i])
    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, match):
    
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    
def match_features_dataset(des_list, match_features):
    
    matches = []
    
    ### START CODE HERE ###
    for i in range(len(des_list)-1):
        matches.append(match_features(des_list[i],des_list[i+1]))
    
    return matches

def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    
    filtered_matches = []
    
    for i in range(len(matches)):
        filtered_matches.append(filter_matches_distance(matches[i], dist_threshold))

    
    return filtered_matches


def filter_matches_dataset_by_fundamental(filter_matches_distance_by_fundamental, matches, kp_list):
    
    filtered_matches = []
    
    for i in range(len(matches)):
        filtered_matches.append(filter_matches_distance_by_fundamental(matches[i], kp_list[i], kp_list[i+1]))
        
    return filtered_matches



def project(points, camera_params, k):
    """Convert 3-D points to 2-D by projecting onto images."""
    
    R = camera_params[:9].reshape(3,3)
    t = camera_params[9:12].reshape(3,1)
    
    P = np.concatenate((R, t), axis=1)
    
    proj = []
    
    for i in range(len(points)):
        projected = k.dot(P).dot(np.array([[points[i][0]], [points[i][1]], [points[i][2]], [1]]))
        proj.append([projected[0][0]/projected[2][0]])
        proj.append([projected[1][0]/projected[2][0]])
        
    return np.asarray(proj)

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, k):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 12]
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    points_proj = project(points_3d, camera_params, k)
    return (points_proj - points_2d.ravel().reshape(len(points_2d)*2,1)).reshape(len(points_2d)*2)


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





def estimate_motion(match, kp1, kp2, k, depth1=None):
    
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

    imagePoints = np.array(image2_points)
    imagePoints = imagePoints.astype('float64')
    objectPoints = np.transpose(objectPoints)
    distCoeffs = np.zeros(4)
    #print(objectPoints.shape)
        
    _, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, k, distCoeffs)
    rmat, _ = cv2.Rodrigues(rvec)
    """ BUNDLE ADJUSTMENT"""
    
    camera_params = np.concatenate((rmat.ravel(), tvec.reshape(3)))


    points_3d = objectPoints
    points_2d = np.asarray(image2_points)
    camera_indices = np.zeros(len(points_2d), dtype=np.int64)
    point_indices = np.arange(0, len(points_2d), dtype=np.int64)
    
    n_cameras = 1 #camera_params.shape[0]
    n_points = points_3d.shape[0]
    
    n = 12 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, k)
    
    
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    
    res = least_squares(fun, x0, jac='3-point', verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, k))
    
    #jac_sparsity=A
    
    X = res.x
    grad = res.grad
    
    rmat = X[:9].reshape(3,3)
    tvec = X[9:12].reshape(3,1)
    
    
    return rmat, tvec, image1_points, image2_points

def get_quaternions(r):
    tr = r[0][0] + r[1][1] + r[2][2]

    if (tr > 0):
        S = math.sqrt(tr+1.0) * 2 
        qw = 0.25 * S
        qx = (r[2][1] - r[1][2]) / S
        qy = (r[0][2] - r[2][0]) / S 
        qz = (r[1][0] - r[0][1]) / S
        
    elif ((r[0][0] > r[1][1]) and (r[0][0] > r[2][2])):
        S = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        qw = (r[2][1] - r[1][2]) / S
        qx = 0.25 * S
        qy = (r[0][1] + r[1][0]) / S 
        qz = (r[0][2] + r[2][0]) / S
        
    elif (r[1][1] > r[2][2]):
        S = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        qw = (r[0][2] - r[2][0]) / S
        qx = (r[0][1] + r[1][0]) / S
        qy = 0.25 * S
        qz = (r[1][2] + r[2][1]) / S
        
    else:
        S = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        qw = (r[1][0] - r[0][1]) / S
        qx = (r[0][2] + r[2][0]) / S
        qy = (r[1][2] + r[2][1]) / S
        qz = 0.25 * S
        
    return np.array([qw, qx, qy, qz])

def get_rotation_from_quaternion(qw, qx, qy, qz):
    R = np.array([[1-(2*qz**2)-(2*qy**2), (2*qy*qx)-(2*qz*qw), (2*qy*qw)+(2*qz*qx)],
                  [(2*qx*qy)+(2*qw*qz), 1-(2*qz**2)-(2*qx**2), (2*qz*qy)-(2*qx*qw)],
                  [(2*qx*qz)-(2*qw*qy), (2*qy*qz)+(2*qw*qx), 1-(2*qy**2)-(2*qx**2)]])
    
    return R


def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    
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

def ned2cam(traj):
    '''
    transfer a ned traj to camera frame traj
    '''
    T = np.array([[0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32) 
    T_inv = np.linalg.inv(T)
    new_traj = []
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))

    for tt in traj_ses:
        ttt=T.dot(tt).dot(T_inv)
        new_traj.append(tf.SE2pos_quat(ttt))
        
    return np.array(new_traj)


def shift0(traj): 
    '''
    Traj: a list of [t + quat]
    Return: translate and rotate the traj
    '''
    traj_ses = tf.pos_quats2SE_matrices(np.array(traj))
    traj_init = traj_ses[0]
    traj_init_inv = np.linalg.inv(traj_init)
    new_traj = []
    for tt in traj_ses:
        ttt=traj_init_inv.dot(tt)
        new_traj.append(tf.SE2pos_quat(ttt))
    return np.array(new_traj)


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
            YX_plt.plot(tr[u,1], tr[u,0], ".-", linewidth=2, markersize=4, zorder=0)
            # YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
            # YX_plt.scatter([0], [0], s=2, c="red", label="Start location", zorder=2)
            ZY_plt.plot(tr[u,2], tr[u,1], ".-", linewidth=2, markersize=4, zorder=0)
            # ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
            # ZY_plt.scatter([0], [0], s=2, c="red", label="Start location", zorder=2)
        elif u==1:
            D3_plt.plot3D(tr[u,0], tr[u,1], tr[u,2], zorder=0,color = 'purple')
            traj_main_plt.plot( tr[u,2], tr[u,0], ".-", label="Trajectory gt", zorder=1, linewidth=1, markersize=1)
            YX_plt.plot(tr[u,1], tr[u,0], ".-", linewidth=1, markersize=1, zorder=0)
            # YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
            # YX_plt.scatter([0], [0], s=2, c="red", label="Start location", zorder=2)
            ZY_plt.plot(tr[u,2], tr[u,1], ".-", linewidth=1, markersize=1, zorder=0)
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


def quatHProd(p, q):
    """Compute the Hamilton product of quaternions `p` and `q`."""
    r = np.array([p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
                  p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
                  p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
                  p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]])
    return r