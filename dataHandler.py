#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:52:53 2020

@author: prithvi
"""

import os
import numpy as np
import cv2

class DatasetHandler:

    def __init__(self,pwd):

        # Set up paths
        root_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.image_right_dir = os.path.join(root_dir_path, pwd,'image_right/')
        self.depth_right_dir = os.path.join(root_dir_path, pwd,'depth_right/')
        self.seg_right_dir = os.path.join(root_dir_path, pwd,'seg_right/')
        self.image_left_dir = os.path.join(root_dir_path, pwd,'image_left/')
        self.depth_left_dir = os.path.join(root_dir_path,pwd ,'depth_left/')
        self.seg_left_dir = os.path.join(root_dir_path, pwd,'seg_left/')
        self.flow_dir = os.path.join(root_dir_path, pwd,'flow/')
        self.pose_left_dir = os.path.join(root_dir_path, pwd,'pose_left.txt')
        self.pose_right_dir = os.path.join(root_dir_path, pwd,'pose_right.txt')

        # Set up data holders
        self.image_right = sorted(os.listdir(self.image_right_dir))
        self.depth_right = sorted(os.listdir(self.depth_right_dir))
        self.seg_right = sorted(os.listdir(self.seg_right_dir))
        self.image_left = sorted(os.listdir(self.image_left_dir))
        self.depth_left = sorted(os.listdir(self.depth_left_dir))
        self.seg_left = sorted(os.listdir(self.seg_left_dir))
        self.flow = sorted(os.listdir(self.flow_dir))
        

        self.k = np.array([[320, 0, 320],
                           [0, 320, 240],
                           [0,   0,   1]], dtype=np.float32)
        

        # Read first frame
        
    def _read_depth_right(self,i):
        depth_right = np.load(self.depth_right_dir + self.depth_right[i])
        return depth_right

    def _read_image_right(self,i):
        image_right = cv2.imread(self.image_right_dir + self.image_right[i])
        return image_right

    def _read_depth_left(self,i):
        depth_left = np.load(self.depth_left_dir + self.depth_left[i])
        return depth_left

    def _read_image_left(self,i):
        image_left = cv2.imread(self.image_left_dir + self.image_left[i])
        return image_left

    def _read_seg_right(self,i):
        seg_right = np.load(self.seg_right_dir + self.seg_right[i])
        return seg_right
    
    def _read_seg_left(self,i):
        seg_left = np.load(self.seg_left_dir + self.seg_left[i])
        return seg_left
    
    def _read_flow(self,i):
        flow =  np.load(self.flow_dir + self.flow[i])
        return flow
    
    def _read_pose_left(self):
        pose = np.loadtxt(self.pose_left_dir)
        return pose
    
    def _read_pose_right(self):
        pose = np.loadtxt(self.pose_right_dir)
        return pose
    
    def _total_files(self):
        path, dirs, files = next(os.walk(self.image_right_dir))
        return len(files)