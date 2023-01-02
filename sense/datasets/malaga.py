import os
import numpy as np


def load_pose(path):
    poses = []
    with open(path, 'r') as calib_file:
        lines = calib_file.readlines()[1:]
        for line in lines:
            items = [float(x) for x in line.split("     ")]
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

def find_closest_time_index(imu_list, time):
    return min(range(len(imu_list)), key=lambda i: abs(imu_list[i][0]-float(time)))
    
def calc_pose_diff(imu_list, cur_time, next_time):
    cur_idx = find_closest_time_index(imu_list, cur_time)
    nxt_idx = find_closest_time_index(imu_list, next_time)
    imu_np = np.array(imu_list)
    position = np.array([0., 0., 0.])
    angle = np.array([0., 0., 0.])
    for i in range(cur_idx - 1, nxt_idx):
        delta_t = 0.1
        angle +=  imu_np[i + 1][4:7] * delta_t
        position = imu_np[i + 1][0:3] * delta_t * delta_t
    return np.concatenate((position, angle))

def malaga_data_helper(path, train_sequences):
    malaga_train = []
    malaga_test = []
    
    #base_dir = os.path.join(path, "malaga")
    base_dir = path + "/malaga"
    sequence_list = os.listdir(base_dir)
    for seq in sequence_list:
        # calib = load_calib(os.path.join(path, seq, "camera_params_rectified_a=0_1024x768.txt"))
        # image_list = os.listdir(os.path.join(base_dir, seq, seq + "_rectified_1024x768_Images"))
        image_list = os.listdir(base_dir + "/" + seq + "/" + seq + "_rectified_1024x768_Images")
        # pose_list = load_pose(os.path.join(base_dir, seq, seq + "_all-sensors_IMU.txt"))
        pose_list = load_pose(base_dir + "/" + seq + "/" + seq + "_all-sensors_IMU.txt")
        image_list.sort()
        for j in range(0, len(image_list) - 20, 2):
            sequence = []
            pose = np.empty([6], dtype = float)
            for k in range(0, 10, 2):
                # cur_left = os.path.join(base_dir, seq, seq + "_rectified_1024x768_Images", image_list[j + k])
                cur_left = base_dir + "/" + seq + "/" + seq + "_rectified_1024x768_Images/" + image_list[j + k]
                # cur_right = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j + k + 1])
                # nxt_left = os.path.join(base_dir, seq, seq + "_rectified_1024x768_Images", image_list[j + k + 2])
                nxt_left = base_dir + "/" + seq + "/" + seq + "_rectified_1024x768_Images/" +image_list[j + k + 2]
                # nxt_right = os.path.join(path, seq, seq + "_rectified_1024x768_Images", image_list[j + k + 3])
                sequence.append([cur_left, nxt_left])
                pose += calc_pose_diff(pose_list, image_list[j + k].split("_")[2], image_list[j + k + 2].split("_")[2])
            sequence.append(pose)
            sequence.append("M")
            malaga_train.append(sequence) if int(seq.split("-")[-1]) in train_sequences else malaga_test.append(sequence)
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