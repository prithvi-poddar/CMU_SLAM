#!/usr/bin/env python
# coding: utf-8

# In[1]:



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




# In[3]:


dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")


# In[6]:


image = dataset_handler._read_image_left(0)

#plt.figure(figsize=(8, 6), dpi=100)
#plt.imshow(image, cmap='gray')


# In[8]:


K = dataset_handler.k



# In[9]:


def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    
    return kp, des


# In[10]:


i = 0
image = dataset_handler._read_image_left(0)
kp, des = extract_features(image)


# In[11]:


def visualize_features(image, kp):
    display = cv2.drawKeypoints(image, kp, None)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)


# In[12]:


i = 0
image = dataset_handler._read_image_left(0)

#visualize_features(image, kp)


# In[13]:


def extract_features_dataset(images, extract_features_function):
   
    kp_list = []
    des_list = []
    
    ### START CODE HERE ###
    for i in images:
        kp, des = extract_features_function(i)
        kp_list.append(kp)
        des_list.append(des)

    return kp_list, des_list


# In[14]:


num_images = dataset_handler._total_files()
images = []
for i in range(num_images):
    images.append(dataset_handler._read_image_left(i))
    
depth_maps = []
for i in range(num_images):
    depth_maps.append(dataset_handler._read_depth_left(i))


# In[15]:


kp_list, des_list = extract_features_dataset(images, extract_features)

i = 0
print("Number of features detected in frame {0}: {1}".format(i, len(kp_list[i])))
print("Coordinates of the first keypoint in frame {0}: {1}\n".format(i, str(kp_list[i][0].pt)))

# Remember that the length of the returned by dataset_handler lists should be the same as the length of the image array
print("Length of images array: {0}".format(len(kp_list)))


# In[16]:


def match_features(des1, des2):
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    
    return matches


# In[17]:


i = 0 
des1 = des_list[i]
des2 = des_list[i+1]

match = match_features(des1, des2)
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(match)))


# In[18]:


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


# In[19]:


def visualize_matches(image1, kp1, image2, kp2, match):
    
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


# In[29]:

"""
n = None
filtering = True

i = 0 
image1 = images[i]
image2 = images[i+1]

kp1 = kp_list[i]
kp2 = kp_list[i+1]

des1 = des_list[i]
des2 = des_list[i+1]

match = match_features(des1, des2)
if filtering:
    dist_threshold = 0.2
    match = filter_matches_distance(match, dist_threshold)

image_matches = visualize_matches(image1, kp1, image2, kp2, match[:n])  
"""

# In[30]:


def match_features_dataset(des_list, match_features):
    
    matches = []
    
    ### START CODE HERE ###
    for i in range(len(des_list)-1):
        matches.append(match_features(des_list[i],des_list[i+1]))


    
    ### END CODE HERE ###
    
    return matches


# In[31]:


matches = match_features_dataset(des_list, match_features)
i = 0
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(matches[i])))
print(len(matches))


# In[32]:


def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    
    filtered_matches = []
    
    for i in range(len(matches)):
        filtered_matches.append(filter_matches_distance(matches[i], dist_threshold))

    
    return filtered_matches


# In[33]:


dist_threshold = 0.2

filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)

if len(filtered_matches) > 0:
    
    # Make sure that this variable is set to True if you want to use filtered matches further in your assignment
    is_main_filtered_m = True
    if is_main_filtered_m: 
        matches = filtered_matches
        
    i = 0
    print("Number of filtered matches in frames {0} and {1}: {2}".format(i, i+1, len(filtered_matches[i])))
    print(len(filtered_matches))


# In[34]:


def estimate_motion(match, kp1, kp2, k, depth1=None):
    
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
    ### START CODE HERE ###
    count=0
    
    
    for m in match:
            
        image1_x, image1_y = kp1[m.queryIdx].pt
        image1_x = int(image1_x)
        image1_y = int(image1_y)
        depth = depth1[image1_y, image1_x]

        if(depth<15):
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

    imagePoints = np.array(image2_points)
    imagePoints = imagePoints.astype('float64')
    objectPoints = np.transpose(objectPoints)
    distCoeffs = np.zeros(4)
    #print(objectPoints.shape)
        
    _, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, k, distCoeffs)
    rmat, _ = cv2.Rodrigues(rvec)
    
    return rmat, tvec, image1_points, image2_points


# In[35]:


i = 0
match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i+1]
k = dataset_handler.k
depth = depth_maps[i]

rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth1=depth)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))


# In[36]:



# In[37]:

"""
i=0
image1  = images[i]
image2 = images[i + 1]

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)
"""

# In[38]:


def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    
    trajectory = [np.array([0, 0, 0])]
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
        

    trajectory = np.array(trajectory).T
    print(P)
    
    return trajectory


# In[39]:


trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)

i = 1
print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))

# Remember that the length of the returned by trajectory should be the same as the length of the image array
print("Length of trajectory: {0}".format(trajectory.shape[1]))


# In[40]:





# In[41]:



# In[ ]:

gt = np.loadtxt("japanesealley_sample_P007/P007/pose_left.txt")


import transformation as tf
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

gt1 = shift0(gt)

gt_final = ned2cam(gt1)

gt_x = gt_final[:,0]
gt_y = gt_final[:, 1]
gt_z = gt_final[:, 2]

true = np.stack((gt_x, gt_y, gt_z), axis=0)
visualize_trajectory([np.array(trajectory),np.array(true)])



