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
import json
import open3d as o3d
dataset_handler = DatasetHandler("japanesealley_sample_P007/P007/")

gt = np.loadtxt("japanesealley_sample_P007/P007/pose_left.txt")
k = dataset_handler.k
num_images = dataset_handler._total_files()


pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
pinhole_camera_intrinsic.set_intrinsics(640, 480, 320.0, 320.0, 320.0, 240.0)

trajectory = [np.array([0, 0, 0])]

P = np.eye(4)

for i in range(num_images-1):
    source_color = o3d.io.read_image("images/img"+str(i)+".png")
    source_depth = o3d.io.read_image("depths/depth"+str(i)+".png")
    target_color = o3d.io.read_image("images/img"+str(i+1)+".png")
    target_depth = o3d.io.read_image("depths/depth"+str(i+1)+".png")
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_color, target_depth)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd_image, pinhole_camera_intrinsic)
    
    option = o3d.odometry.OdometryOption()
    odo_init = np.identity(4)
    
    
    
    [success_color_term, trans_color_term, info] = o3d.odometry.compute_rgbd_odometry(
             source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
             odo_init, o3d.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term, info] = o3d.odometry.compute_rgbd_odometry(
             source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
             odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    
    rmat = trans_hybrid_term[:3,:3]
    t = trans_hybrid_term[:3,3]
    
    R = rmat
    #t = np.array([tvec[0,0],tvec[1,0],tvec[2,0]])
    P_new = np.eye(4)
    P_new[0:3,0:3] = R.T
    P_new[0:3,3] = (-R.T).dot(t)
    P = P.dot(P_new)
    trajectory.append(P[:3,3])
    print(i)
trajectory = np.array(trajectory).T
    
gt1 = shift0(gt)

gt_final = ned2cam(gt1)

gt_x = gt_final[:,0]
gt_y = gt_final[:, 1]
gt_z = gt_final[:, 2]

true = np.stack((gt_x, gt_y, gt_z), axis=0)
visualize_trajectory([np.array(trajectory),np.array(true)])

    










"""

img1 = dataset_handler._read_image_left(0)
img2 = dataset_handler._read_image_left(1)
img3 = dataset_handler._read_image_left(2)

depth1 = dataset_handler._read_depth_left(0)
depth2 = dataset_handler._read_depth_left(1)
depth3 = dataset_handler._read_depth_left(2)

kp1, des1 = extract_features(img1)
kp2, des2 = extract_features(img2)
kp3, des3 = extract_features(img3)

match12 = match_features(des1, des2)
match23 = match_features(des2, des3)
match13 = match_features(des1, des3)

filtered_match12 = filter_matches_distance(match12, 0.3)
filtered_match23 = filter_matches_distance(match23, 0.3)
filtered_match13 = filter_matches_distance(match13, 0.2)

match12 = filtered_match12
match23 = filtered_match23
match13 = filtered_match13

image1_points = []
image2_points = []


count = 0
for m in match12:
    image1_x, image1_y = kp1[m.queryIdx].pt
    depth = depth1[int(image1_y), int(image1_x)]
    image1_points.append([image1_x, image1_y])
    image1_3D = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]).reshape([3,1]))
    
    if count==0:
        objectPoints = image1_3D
    else:
        objectPoints = np.c_[objectPoints, image1_3D]
        
    image2_x, image2_y = kp2[m.trainIdx].pt
    image2_points.append([image2_x, image2_y])
    count+=1
    
imagePoints12 = np.array(image2_points, dtype=np.float64)
objectPoints12 = np.transpose(objectPoints)
distCoeffs = np.zeros(4)

_, rvec12, tvec12, inliers12 = cv2.solvePnPRansac(objectPoints12, imagePoints12, k, distCoeffs)


image1_points = []
image2_points = []

count = 0
for m in match23:
    image1_x, image1_y = kp2[m.queryIdx].pt
    depth = depth2[int(image1_y), int(image1_x)]
    image1_points.append([image1_x, image1_y])
    image1_3D = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]).reshape([3,1]))
    
    if count==0:
        objectPoints = image1_3D
    else:
        objectPoints = np.c_[objectPoints, image1_3D]
        
    image2_x, image2_y = kp3[m.trainIdx].pt
    image2_points.append([image2_x, image2_y])
    count+=1
    
imagePoints23 = np.array(image2_points, dtype=np.float64)
objectPoints23 = np.transpose(objectPoints)
distCoeffs = np.zeros(4)

_, rvec23, tvec23, inliers23 = cv2.solvePnPRansac(objectPoints23, imagePoints23, k, distCoeffs)

R12, _ = cv2.Rodrigues(rvec12)
R23, _ = cv2.Rodrigues(rvec23)

P12 = np.eye(4)
P23 = np.eye(4)

P12[:3,:3] = R12
P12[:3,3] = tvec12.reshape(3)

P23[:3,:3] = R23
P23[:3,3] = tvec23.reshape(3)

P13 = P23.dot(P12)

R13 = P13[:3,:3]
tvec13 = P13[:3,3].reshape(3,1)
rvec13, _ = cv2.Rodrigues(R13)

image1_points = []
image2_points = []

count = 0
for m in match13:
    image1_x, image1_y = kp1[m.queryIdx].pt
    depth = depth1[int(image1_y), int(image1_x)]
    image1_points.append([image1_x, image1_y])
    image1_3D = np.dot(np.linalg.inv(k), np.array([image1_x*depth, image1_y*depth, depth]).reshape([3,1]))
    
    if count==0:
        objectPoints = image1_3D
    else:
        objectPoints = np.c_[objectPoints, image1_3D]
        
    image2_x, image2_y = kp3[m.trainIdx].pt
    image2_points.append([image2_x, image2_y])
    count+=1
    
imagePoints13 = np.array(image2_points, dtype=np.float64)
objectPoints13 = np.transpose(objectPoints)
distCoeffs = np.zeros(4)

_, rvec_, tvec_, inliers13 = cv2.solvePnPRansac(objectPoints13, imagePoints13, k, distCoeffs)

new_3d=[]
new_2d=[]
for i in inliers13:
    new_3d.append(objectPoints13[i])
    new_2d.append(imagePoints13[i])
    
new_3d=np.asarray(new_3d).reshape(len(new_3d),3)
new_2d=np.asarray(new_2d).reshape(len(new_2d),2)

def fun(camera_params, k, points_2d, points_3d):
    rvec = np.array([camera_params[0],camera_params[1], camera_params[2]]).reshape(3,1)
    tvec = np.array([camera_params[3],camera_params[4], camera_params[5]]).reshape(3,1)
    distCoeffs = np.zeros(4)
    proj,_ = cv2.projectPoints(points_3d, rvec, tvec, k, distCoeffs)
    proj = proj.reshape(len(proj),2)
    loss = []
    for i in range(len(proj)):
        dist = np.sqrt((proj[i][0]-points_2d[i][0])**2 + (proj[i][1]-points_2d[i][1])**2)
        loss.append(dist)
    #return np.asarray(loss)
    return(proj.ravel()-points_2d.ravel())


points_3d = new_3d
points_2d = new_2d
#camera_params = np.array([rvec_[0][0], rvec_[1][0], rvec_[2][0], tvec_[0][0], tvec_[1][0], tvec_[2][0]])

camera_params = np.array([rvec13[0][0], rvec13[1][0], rvec13[2][0], tvec13[0][0], tvec13[1][0], tvec13[2][0]])
x0 = camera_params.ravel()
f0 = fun(x0, k, points_2d, points_3d)

#plt.plot(f0)

res = least_squares(fun, x0, verbose=2, jac='3-point', ftol=1e-15, method='trf', tr_solver='exact', loss='cauchy', args=(k, points_2d, points_3d))
rvec13 = res.x[:3].reshape(3,1)
tvec13 = res.x[3:].reshape(3,1)


"""











