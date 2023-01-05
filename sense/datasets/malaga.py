import os
import numpy as np
from datetime import datetime

flag = True
last_idx = 0

def load_pose(path):
    poses = []
    with open(path, 'r') as calib_file:
        lines = calib_file.readlines()[1:]
        for line in lines:
            items = [float(x) for x in line.split("    ")]
            poses.append(items[:7])
    return poses

# [fx  s  cx]
# [0   fy cy]
# [0   0   1]  
def load_calib(path):
    def parse_calib_matrix(lines):
        matrix = np.identity(3)
        for line in lines:
            if "=" in line:
                key, value = line.split("=")
                if key == 'cx':
                    matrix[0, 2] = float(value)
                elif key == 'cy':
                    matrix[1, 2] = float(value)
                elif key == 'fx':
                    matrix[0, 0] = float(value)
                elif key == 'fy':
                    matrix[1, 1] = float(value)
        return matrix
    
    with open(path, 'r') as calib_file:
        lines = calib_file.readlines()
        CL = parse_calib_matrix(lines[:12])
        CR = parse_calib_matrix(lines[13:])
    return CL, CR

def load_image_list(path):
    with open(path, 'r') as file:
        return file.readlines()

def find_closest_time_index(imu_list, time, idx):
    global flag
    diff = -1.
    best = -1
    for i in range(idx, len(imu_list)):
        if abs(float(imu_list[i][0]) - float(time)) <= 0.1:
            diff = abs(float(imu_list[i][0]) - float(time))
            best = i
            for j in range(1, 6):
                if i + j < len(imu_list):
                    if abs(float(imu_list[i + j][0]) - float(time)) < diff:
                        diff = abs(float(imu_list[i + j][0]) - float(time))
                        best = i + j
            return best
    flag = False
    return best

def calc_pose_diff(imu_np, cur_time, next_time):
    global last_idx, flag
    cur_idx = find_closest_time_index(imu_np, cur_time, last_idx)
    if cur_idx > 0:
        if cur_idx - 10 > 0:
            last_idx = cur_idx - 10
    else:
        return np.empty([6], dtype = float)
    nxt_idx = find_closest_time_index(imu_np, next_time, cur_idx)
    position = np.array([0., 0., 0.])
    angle = np.array([0., 0., 0.])
    for i in range(cur_idx, nxt_idx):
        delta_t = 0.05
        angle +=  imu_np[i][4:7] * delta_t
        position += imu_np[i][1:4] * delta_t * delta_t
    return np.concatenate((position, angle))

def malaga_data_helper(path, train_sequences):
    global flag, last_idx, ls
    malaga_train = []
    malaga_test = []
    
    base_dir = os.path.join(path, "malaga").replace("\\","/")
    sequence_list = os.listdir(base_dir)
    for seq in sequence_list:
        last_idx = 0
        i = 0
        t = datetime.now()
        # calib = load_calib(os.path.join(path, seq, "camera_params_rectified_a=0_1024x768.txt"))
        image_list = os.listdir(os.path.join(base_dir, seq, seq + "_rectified_1024x768_Images"))
        pose_list = load_pose(os.path.join(base_dir, seq, seq + "_all-sensors_IMU.txt"))
        image_list.sort()
        imu_np = np.array(pose_list)
        print(seq)
        for j in range(0, len(image_list) - 20, 2):
            flag = True
            sequence = []
            pose = np.empty([6], dtype = float)
            for k in range(0, 10, 2):
                if not flag:
                    break
                cur_left = os.path.join(base_dir, seq, seq + "_rectified_1024x768_Images", image_list[j + k]).replace("\\","/")
                # cur_right = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j + k + 1])
                nxt_left = os.path.join(base_dir, seq, seq + "_rectified_1024x768_Images", image_list[j + k + 2]).replace("\\","/")
                # nxt_right = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j + k + 3])
                sequence.append([cur_left, nxt_left])
                pose += calc_pose_diff(imu_np, image_list[j + k].split("_")[2], image_list[j + k + 2].split("_")[2])   
            if flag:
                i += 1
                sequence.append(pose)
                malaga_train.append(sequence) if int(seq.split("-")[-1]) in train_sequences else malaga_test.append(sequence)
        print(f"{str(datetime.now() - t)}")
        print(f"{i}/{len(image_list) / 2}")
    return malaga_train, malaga_test

def malaga_flow_data_helper(path, train_sequences):
    malaga_train = []
    malaga_test = []
    
    sequence_list = os.listdir(path)
    for seq in sequence_list:
        calib = load_calib(os.path.join(path, seq, "camera_params_rectified_a=0_1024x768.txt"))
        image_list = os.listdir(os.path.join(path, seq, seq + "_rectified_1024x768_Images"))
        image_list.sort()
        for j in range(0, len(image_list) - 4, 2):
            cur_left = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j])
            cur_right = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j + 1])
            nxt_left = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j + 2])
            nxt_right = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j + 3])
            item = [cur_left, nxt_left]
            malaga_train.append(item) if int(seq.split("-")[-1]) in train_sequences else malaga_test.append(item)
    return malaga_train, malaga_test