import os
import cv2
import torch
import numpy as np
from math import atan2, sqrt, asin, cos, pi
from scipy.spatial.transform import Rotation

def rotation_matrix_to_euler_angles(rotation):
    if rotation[2,0] != 1 and rotation[2,0] != -1:
        pitch = -asin(rotation[2,0])
        roll = atan2(rotation[2,1]/cos(pitch), rotation[2,2]/cos(pitch))
        yaw = atan2(rotation[1,0]/cos(pitch), rotation[0,0]/cos(pitch))
    else:
        yaw = 0
        if rotation[2,0] == -1:
            pitch = pi/2
            roll = yaw + atan2(rotation[0,1], rotation[0,2])
        else:
            pitch = -pi/2
            roll = -yaw + atan2(-rotation[0,1], -rotation[0,2])

    return np.array([yaw, pitch, roll])

def matrix_to_pose(pose):
    pose = np.array(pose)
    # Convert array to 3x4 numpy array
    # Convert the 3x3 rotation matrices to Euler angles
    rotations = pose.reshape((3, 4))
    angles = rotation_matrix_to_euler_angles(rotations[:, :3])

    # Combine the translation and Euler angles to form a 6-dimensional pose vector
    poses_6d = np.concatenate([rotations[:, -1], angles])

    return poses_6d

def load_pose_file(path):
    # Load the poses file
    with open(path, 'r') as f:
        poses = f.readlines()

    # Extract the relative pose transformations between consecutive frames
    relative_poses = []
    for i in range(len(poses) - 1):
        pose1 = np.array(poses[i].strip().split()).astype(np.float32).reshape(3, 4)
        pose1 = np.vstack([pose1, [0, 0, 0, 1]])  # Append a row to make pose1 a 4x4 matrix
        pose2 = np.array(poses[i+1].strip().split()).astype(np.float32).reshape(3, 4)
        pose2 = np.vstack([pose2, [0, 0, 0, 1]])  # Append a row to make pose2 a 4x4 matrix
        relative_pose = np.linalg.inv(pose1) @ pose2
        relative_poses.append(relative_pose)

    # Convert the relative pose transformations to the 6-dimensional pose format
    poses_6d = []
    for i in range(len(relative_poses)):
        rotation = rotation_matrix_to_euler_angles(relative_poses[i][:3, :3])
        translation = relative_poses[i][:3, 3]
        pose_6d = np.concatenate([translation, rotation.ravel()])
        poses_6d.append(pose_6d)
    return poses_6d


def load_pose(path):
    poses = {}
    # Iterate over the pose files in the dataset
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith('.txt'):
                # Load the pose data from the current file
                with open(os.path.join(dirpath, f), 'r') as pose_file:
                    lines = pose_file.readlines()
                    pose = []
                    for line in lines:
                        pose.append([float(x) for x in line.split()])
                # Add the pose data to the list
                poses[f.split(".")[0]] = pose
    return poses

def calc_pose_diff(cur_pose, next_pose):
    return (matrix_to_pose(next_pose) - matrix_to_pose(cur_pose)).tolist()

def get_calib_matrix(calib):
    calib = np.array(calib)
    calib = calib.reshape(3,4)
    return calib[:, :3] 
    
def load_calib(path):
    calib = []
    # Load the calibration data from the file
    with open(os.path.join(path, 'calib.txt'), 'r') as calib_file:
        lines = calib_file.readlines()
        for line in lines:
            if 'P0' in line:
                # Parse the calibration data for the left camera
                calib_left = [float(x) for x in line.split()[1:]]
                calib.append(get_calib_matrix(calib_left))
            elif 'P1' in line:
                # Parse the calibration data for the right camera
                calib_right = [float(x) for x in line.split()[1:]]
                calib.append(get_calib_matrix(calib_right))
    return calib
                
def kitti_vo_data_helper(path, train_sequences, val_sequences):
    kitti_vo_train = []
    kitti_vo_test = []
    
    base_dir = os.path.join(path, "kitti_vo", "dataset").replace("\\","/")
    pose_list = load_pose(os.path.join(base_dir, "poses").replace("\\","/"))
    for i in range(len(pose_list.keys())):
        poses = load_pose_file(os.path.join(base_dir, "poses", f"{i:02}.txt").replace("\\","/"))
        # calib = load_calib(os.path.join(base_dir, "sequences", f"{i:02}"))
        left_img_list = os.listdir(os.path.join(base_dir, "sequences", f"{i:02}", "image_2").replace("\\","/"))
        left_img_list.sort()
        # right_img_list = os.listdir(os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3"))
        # right_img_list.sort()
        for j in range(len(left_img_list) - 5):
            sequence = []
            pose = np.zeros(6)
            for k in range(5):
                cur_left = os.path.join(base_dir, "sequences", f"{i:02}", "image_2", left_img_list[j + k]).replace("\\","/")
                # cur_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j + k])
                nxt_left = os.path.join(base_dir, "sequences", f"{i:02}", "image_2", left_img_list[j + k + 1]).replace("\\","/")
                # nxt_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j + k + 1])
                sequence.append([cur_left, nxt_left])
                pose += poses[j + k]
            sequence.append(pose)
            # sequence.append(calib)
            if i in train_sequences:
                kitti_vo_train.append(sequence)  
            else:
                if i in val_sequences:
                    kitti_vo_test.append(sequence)
    return kitti_vo_train, kitti_vo_test

def kitti_vo_flow_data_helper(path, train_sequences):
    kitti_vo_train = []
    kitti_vo_test = []
    seq_list = os.listdir(os.path.join(path, "dataset", "sequences").replace("\\","/"))
    for i in range(len(seq_list)):
        calib = load_calib(os.path.join(path, "dataset", "sequences", f"{i:02}").replace("\\","/"))
        left_img_list = os.listdir(os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2").replace("\\","/"))
        left_img_list.sort()
        right_img_list = os.listdir(os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3").replace("\\","/"))
        right_img_list.sort()
        for j in range(len(left_img_list) - 1):
            cur_left = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2", left_img_list[j]).replace("\\","/")
            cur_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j]).replace("\\","/")
            nxt_left = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2", left_img_list[j + 1]).replace("\\","/")
            nxt_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j + 1]).replace("\\","/")
            item = [cur_left, nxt_left]
            kitti_vo_train.append(item) if i in train_sequences else kitti_vo_test.append(item)
    return kitti_vo_train, kitti_vo_test