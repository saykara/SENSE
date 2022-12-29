import os
import cv2
import torch
import numpy as np
from scipy.spatial.transform import euler_from_matrix

def matrix_to_pose(pose):
    # Convert array to 3x4 numpy array
    matrix = np.array(pose)
    matrix = matrix.reshape(3, 4)
    
    # Extract the translation vector and rotation matrix from the 3x4 matrix
    translation = matrix[:3, 3]
    rotation = matrix[:3, :3]

    # # Convert the rotation matrix to a quaternion representation
    # quat = np.zeros(4)
    # quat[0] = np.sqrt(1 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2]) / 2
    # quat[1] = (rotation[2, 1] - rotation[1, 2]) / (4 * quat[0])
    # quat[2] = (rotation[0, 2] - rotation[2, 0]) / (4 * quat[0])
    # quat[3] = (rotation[1, 0] - rotation[0, 1]) / (4 * quat[0])

    yaw, pitch, roll = euler_from_matrix(rotation, 'xyz')
    # Concatenate the translation and quaternion to form the 6-dimensional pose
    return np.concatenate((translation, yaw, pitch, roll))

def load_pose(path):
    poses = []
    # Iterate over the pose files in the dataset
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith('.txt'):
                # Load the pose data from the current file
                with open(os.path.join(dirpath, f), 'r') as pose_file:
                    pose = [float(x) for x in pose_file.readline().split()]
                # Add the pose data to the list
                poses.append(pose)
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
                
def kitti_vo_data_helper(path, train_sequences):
    kitti_vo_train = []
    kitti_vo_test = []
    pose_list = load_pose(os.path.join(path, "dataset", "poses"))
    for i in range(len(pose_list)):
        calib = load_calib(os.path.join(path, "dataset", "sequences", f"{i:02}"))
        left_img_list = os.listdir(os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2"))
        right_img_list = os.listdir(os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3"))
        for j in range(len(left_img_list) - 6):
            sequence = []
            for k in range(5):
                cur_left = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2", left_img_list[j + k])
                cur_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j + k])
                nxt_left = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2", left_img_list[j + k + 1])
                nxt_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j + k + 1])
                sequence.append([cur_left, cur_right, nxt_left, nxt_right, calc_pose_diff(pose_list[i][j + k], pose_list[i][j + k + 1])])
            sequence.append(calib)
            kitti_vo_train.append(sequence) if i in train_sequences else kitti_vo_test.append(sequence)
    return kitti_vo_train, kitti_vo_test

def kitti_vo_flow_data_helper(path, train_sequences):
    kitti_vo_train = []
    kitti_vo_test = []
    pose_list = load_pose(os.path.join(path, "dataset", "poses"))
    for i in range(len(pose_list)):
        calib = load_calib(os.path.join(path, "dataset", "sequences", f"{i:02}"))
        left_img_list = os.listdir(os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2"))
        right_img_list = os.listdir(os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3"))
        for j in range(len(left_img_list) - 1):
            cur_left = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2", left_img_list[j ])
            cur_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j])
            nxt_left = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_2", left_img_list[j + 1])
            nxt_right = os.path.join(path, "dataset", "sequences", f"{i:02}", "image_3", right_img_list[j + 1])
            item = [cur_left, cur_right, nxt_left, nxt_right, calib]
            kitti_vo_train.append(item) if i in train_sequences else kitti_vo_test.append(item)
    return kitti_vo_train, kitti_vo_test