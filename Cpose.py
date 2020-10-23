
### AUTHOR : SWADHIN AGRAWAL
### CAMERA POSE ESTIMATION

#   Libraries____________________________________________________________________

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
import transformation as tf
import trajectory_transform as ttfm

#________________________________________________________________________________

###########################################################################################################################
###   For running with downloaded dataset 
###########################################################################################################################

#   Data extraction______________________________________________________________

pwd = "japanesealley_sample_P007/P007/"
dataset_handler = DatasetHandler(pwd)


orb = cv2.ORB_create(nfeatures=3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

k = dataset_handler.k
Y = np.array([[0,-1,0],[1,0,0],[0,0,1]])
Q = np.array([[1,0,0,-k[0,2]],[0,1,0,-k[1,2]],[0,0,0,-k[0,0]],[0,0,-4,0]]) # Baseline = 0.25m

tr_r_gt = dataset_handler._read_pose_right()
tr_l_gt = dataset_handler._read_pose_left()
#________________________________________________________________________________

#   Definations__________________________________________________________________

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def extract_features(image, num_feat_per_grid=10):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a image
    num_feat_per_grid -- number of features to retain from each grid if binning=True (Default--5)

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
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

def filter_matches_distance(kpl,match, dist_threshold):
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
    gspec = gridspec.GridSpec(1, 2)

    mat_plt = plt.subplot(gspec[0, 1])
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

def get_matching(images,kp_list,des_list):
    '''
    Tries matching any two given pair of images until minimum number of matches are obtained
    
    Arguments:
    images -- pair of images
    kp_list -- list of key points for each images
    des_list -- list of descriptions for each set of keypoints

    Returns:
    match -- array of matches between two images
    '''
    flag = 0
    gamma = 1.1
    while flag ==0:
        match = match_features_dataset(des_list, match_features)[0]
        dist_threshold = 0.38
        match = filter_matches_distance(kp_list,match, dist_threshold)
        if len(match)>8:
            flag = 1
        else:
            images = adjust_gamma(images, gamma = gamma)
            kp_list,des_list = extract_features_dataset(images, extract_features)
            gamma +=0.1
        
    return match

def get_F(kp_list,match):
    """
    Calculates Fundamental matrix along with removing outliers using RANSAC

    Arguments:
    kp_list -- list of key points for each images
    match -- list of matches between the key points in kp_list

    Returns: 
    F[0]: 3x3 Fundamental matrix
    """
    r_match = []
    l_match = []
    for matched in match:
        r = np.array(kp_list[0][matched.queryIdx].pt)
        r_match.append(r)
        l = np.array(kp_list[1][matched.trainIdx].pt)
        l_match.append(l)
    F = cv2.findFundamentalMat(np.array(r_match),np.array(l_match),method=cv2.FM_RANSAC)
    while [0] in F[1]:
        de = []
        for j in range(len(F[1])-1,-1,-1):
            if F[1][j]==[0]:
                de.append(j)
        for j in de:
            del match[j]
        r_match = []
        l_match = []
        for matched in match:
            r = np.array(kp_list[0][matched.queryIdx].pt)
            r_match.append(r)
            l = np.array(kp_list[1][matched.trainIdx].pt)
            l_match.append(l)
        F = cv2.findFundamentalMat(np.array(r_match),np.array(l_match),method=cv2.FM_RANSAC)
    return F[0]

def estimate_motion_temp(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    objectPoints -- a list of 3D coordinates of features in current frame
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []

    count=0
    for matcher in match:
        image1_x, image1_y = kp1[matcher.queryIdx].pt
        image1_x = int(image1_x)
        image1_y = int(image1_y)
        depth = depth1[image1_y, image1_x]

        if(depth<50):
            image1_points.append([image1_x, image1_y])
            image1_3D = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]))#.reshape([3,1]))
            # print(image1_3D)
            if count == 0:
                objectPoints = image1_3D
            else:
                objectPoints = np.c_[objectPoints, image1_3D]

            image2_x, image2_y = kp2[matcher.trainIdx].pt
            image2_x = int(image2_x)
            image2_y = int(image2_y)
            image2_points.append([image2_x, image2_y])
            count+=1
        
    imagePoints = np.array(image2_points)
    imagePoints = imagePoints.astype('float64')
    objectPoints = np.transpose(objectPoints)
    objectPoints = objectPoints.astype('float64')
    distCoeffs = np.zeros(4)
        
    _, rvec, tvec,inliers = cv2.solvePnPRansac(objectPoints, imagePoints, k, distCoeffs)
    rmat, _ = cv2.Rodrigues(rvec)
    
    return rmat, tvec, image1_points, image2_points,objectPoints

def estimate_trajectory_temp(estimate_motion, match, kp_list, k,P,depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 
    P -- last frame projection matrix
    
    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location
                  
                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven 
                  at the initialization of this function
    object -- 3D coordinates of features
    P -- Current projection matrix
    """   
    kp1 = kp_list[0]
    kp2 = kp_list[1]

    rmat, tvec, image1_points, image2_points,objects = estimate_motion(match, kp1, kp2, k, depth_maps )
    R = rmat
    t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
    P_new = np.eye(4)
    P_new[0:3,0:3] = R.T
    P_new[:3,3] = (-R.T).dot(t)
    P = P.dot(P_new)
    trajectory = P[:3,3]  
    
    return trajectory,objects,P

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
   

#________________________________________________________________________________

#   Main function________________________________________________________________

plt.ion()
fig = plt.figure(figsize=(16, 6), dpi=100)
pt_cld = []


# gt_r = tf.quat2SO(tr_l_gt[0,3:])
# print(gt_r)
# projection[0][0:3,0:3] = gt_r.T

traj_r_gt = []
tr_l_gt = ttfm.shift0(tr_l_gt)
traj_l_gt = ttfm.ned2cam(tr_l_gt)
trajectory_l =[[tr_l_gt[0,0],tr_l_gt[0,1],tr_l_gt[0,2]]]
projection = [np.eye(4)]
projection[0][0:3,3] = trajectory_l[0]

for i in range(len(dataset_handler.image_right)-1):
    
    images_l = np.array([dataset_handler._read_image_left(i),dataset_handler._read_image_left(i+1)])
    depth_l = dataset_handler._read_depth_left(i)

    
    
    kp_list_l,des_list_l = extract_features_dataset(images_l, extract_features)
    
    match_l = get_matching(images_l,kp_list_l,des_list_l)
    
    F_l = get_F(kp_list_l,match_l)
    
    # visualize_matches(images_l[0],kp_list_l[0],images_l[1],kp_list_l[1],match_l)
    
    traj_l,points,p = estimate_trajectory_temp(estimate_motion_temp, match_l, kp_list_l, k ,projection[i],depth_maps=depth_l)
    projection.append(p)
    trajectory_l.append(traj_l)
    
    for n in points:
        pt_cld.append(np.add(n,traj_l))

    visualize_trajectory([np.array(trajectory_l).T,np.array(traj_l_gt).T])#,np.array(pt_cld).T])
    
    plt.show()
    plt.pause(0.01)
    print(i)











#________________________________________________________________________________________________________________________________________________________________







# count = 0
# for i in range(400):#len(dataset_handler.image_right)-15):
#     images_r = [dataset_handler._read_image_right(i), dataset_handler._read_image_right(i+1)]
#     images_l = [dataset_handler._read_image_left(i), dataset_handler._read_image_left(i+1)]
#     traj_r_gt.append(tr_r_gt[i,:3])
#     traj_l_gt.append(tr_l_gt[i,:3])
#     depth_map_r = [dataset_handler._read_depth_right(i), dataset_handler._read_depth_right(i+1)]
#     depth_map_l = [dataset_handler._read_depth_left(i), dataset_handler._read_depth_left(i+1)]

#     # Part 1. Features Extraction
#     kp_list_r, des_list_r = extract_features_dataset(images_r, extract_features)
#     kp_list_l, des_list_l = extract_features_dataset(images_l, extract_features)
#     # print(type(des_list_l))
#     # print(type(des_list_l[0]))
#     # print(type(des_list_l[1]))
#     if isinstance(des_list_l[0],type(None)) == False and isinstance(des_list_l[1],type(None))==False:
#         # Part II. Feature Matching
#         match_r = match_features(des_list_r[0],des_list_r[1])
#         match_l = match_features(des_list_l[0],des_list_l[1])
#         # match_stereo = match_features(des_list_r[0],des_list_l[0])
#         # matches = [match_r,match_l,match_stereo]
#         matches = [match_r,match_l]
#         # Set to True if you want to use filtered matches or False otherwise
#         is_main_filtered_m = True
#         if is_main_filtered_m:
#             dist_threshold = 0.3
#             filtered_matches = filter_matches_dataset(filter_matches_distance,matches, dist_threshold)
#             matches = filtered_matches
#         # img3 = cv2.drawMatches(images_r[0],kp_list_r[0],images_r[1],kp_list_r[1],matches[0][:50],None)
#         # plt.imshow(img3)
#         # plt.show()
#         # plt.pause(1)
#         # plt.clf()
#         # img3 = cv2.drawMatches(images_l[0],kp_list_l[0],images_l[1],kp_list_l[1],matches[1][:50],None)
#         # plt.imshow(img3)
#         # plt.show()
#         # plt.pause(1)
#         # plt.clf()
#         # Part III. Trajectory Estimation
#         # print(len(matches[0]))
        
#         plt.show()
#         plt.pause(0.01)
# # Visualize camera trajectory

# print(len(np.array(trajectory_r).T))














# def get_stereo_mono_kp(kp_list,match):
#     stereo_kp = []
#     bin = []
#     for matched in match:
#         r = np.array(kp_list[0][matched.queryIdx].pt)
#         l = np.array(kp_list[1][matched.trainIdx].pt)
#         bin.append(matched.trainIdx)
#         stereo_kp.append(np.array([l[0],l[1],r[0]]))
#     mono_kp = []
#     for i in range(len(kp_list[1])):
#         if i not in bin:
#             mono_kp.append(kp_list[1][i].pt)
#     del bin
#     return stereo_kp,mono_kp

# def C_pose(E_U,E_V):
#     t = [E_U[:,2],-1*E_U[:,2]]
#     R = [E_U @ Y @ (E_V.T),E_U @ np.linalg.inv(Y) @ (E_V.T)]
#     if np.linalg.det(R[0])<0:
#         R = -np.array(R)
#         t = -np.array(t)
#     # print(np.linalg.det(R[0]))
#     # print(np.linalg.det(R[1]))
#     pose = []
#     for i in t:
#         for j in R:
#             pose.append([j,i])
#     return pose




#######################################################################
#per frame operations
#######################################################################
# images_r = np.array([dataset_handler._read_image_right(i),dataset_handler._read_image_right(i+1)])
# images_stereo = np.array([images_r[0],images_l[0]])
# traj_r_gt.append(tr_r_gt[i,:3])
    
# qr = tr_l_gt[i,3]
# q1 = tr_l_gt[i,4]
# q2 = tr_l_gt[i,5]
# q3 = tr_l_gt[i,6]
# quat = np.array([qr,q1,q2,q3])
# rot = Rotation.from_quat(quat)
# gt_trans = (rot.as_matrix()).dot(tr_l_gt[i,:3])
# kp_list_stereo,des_list_stereo = extract_features_dataset(images_stereo, extract_features)
# kp_list_r,des_list_r = extract_features_dataset(images_r, extract_features)
#____________________________________________________________________________
#   Feature visualization____________________________________________________
# for i in range(len(kp_list)):    
#     visualize_features(images[i],kp_list[i])
#____________________________________________________________________________
#   Feature matching_________________________________________________________
# match_stereo = get_matching(images_stereo,kp_list_stereo,des_list_stereo)
# match_r = get_matching(images_r,kp_list_r,des_list_r)
#____________________________________________________________________________
#   Fundamental Matrix_______________________________________________________
# F_stereo = get_F(kp_list_stereo,match_stereo)
# F_stereo /= np.linalg.norm(F_stereo)
# F_r = get_F(kp_list_r,match_r)
# F_r /= np.linalg.norm(F_r)
# F_l /= np.linalg.norm(F_l)
#____________________________________________________________________________
#   Essential Matrix_________________________________________________________
# E_stereo = k.T @ F_stereo @ k
# E_stereo = E_stereo/np.linalg.norm(E_stereo)
# E_r = k.T @ F_r @ k
# E_r = E_r/np.linalg.norm(E_r)
# E_l = k.T @ F_l @ k
# E_l = E_l/np.linalg.norm(E_l)
# E_U_stereo,E_D_stereo,E_V_stereo = np.linalg.svd(E_stereo)
# E_U_r,E_D_r,E_V_r = np.linalg.svd(E_r)
# E_U_l,E_D_l,E_V_l = np.linalg.svd(E_l)
#____________________________________________________________________________
#   Visualize Match__________________________________________________________
# visualize_matches(images_stereo[0],kp_list_stereo[0],images_stereo[1],kp_list_stereo[1],match_stereo)
# visualize_matches(images_r[0],kp_list_r[0],images_r[1],kp_list_r[1],match_r)

# plt.pause(0.01)
#____________________________________________________________________________
#   Stereo and Mono Keypoints________________________________________________
# stereo_stereo,mono_stereo = get_stereo_mono_kp(kp_list_stereo,match_stereo)
# stereo_r,mono_r = get_stereo_mono_kp(kp_list_r,match_r)
# stereo_l,mono_l = get_stereo_mono_kp(kp_list_l,match_l)
#____________________________________________________________________________

#   Camera Pose______________________________________________________________
# poses = {"r_cam":[], "l_cam":[],"3d_point_cloud":[]}
# cam_rel_pose_stereo = C_pose(E_U_stereo,E_V_stereo)
# cam_rel_pose_r = C_pose(E_U_r,E_V_r)
# cam_rel_pose_l = C_pose(E_U_l,E_V_l)
#____________________________________________________________________________
#   Point Cloud______________________________________________________________
# disparity_t = 0.25*k[0,0]/stereo.compute(images_l[0][:,:,0],images_r[0][:,:,0])
# disparity_t1 = 0.25*k[0,0]/stereo.compute(images_l[1][:,:,0],images_r[1][:,:,0])
# plt.imshow(disparity_t,"gray")
# plt.show()
# plt.pause(0.1)
# print(trajectory_l)

# print(pt_cld)


# plt.clf()
#____________________________________________________________________________
#   KeyFrame Insertion_______________________________________________________
# covisibility.append({"keyframe":{"cam_pose":[] ,"k_intrinsic": ,"frame_features": kp_list},"edge":})
#____________________________________________________________________________
#   Map Point Insertion______________________________________________________
# map.append({"3Dpose":[] ,"mean_dir":[] ,"Min_Ham_dis_des": ,"vis_range":[]})

    # D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    # D3_plt.set_xlim3d(min, max)
    # D3_plt.set_ylim3d(min, max)
    # D3_plt.set_zlim3d(min, max)