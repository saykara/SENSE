import torch.utils.data as data
import os.path
import cv2
import numpy as np


class VODataset(data.Dataset):
    def __init__(self, path_list, pose_path, transform, seq_len = 5):
        super(VODataset, self).__init__()
        self.path_list = path_list
        self.image_loader = self.imread
        self.pose_list = self.read_poses(pose_path)
        self.transform = transform
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.path_list) - self.seq_len - 1

    def __getitem__(self, index):
        cur_l = self.image_loader(self.path_list[0][index : index + self.seq_len])
        cur_r = self.image_loader(self.path_list[1][index : index + self.seq_len])
        nxt_l = self.image_loader(self.path_list[2][index : index + self.seq_len])
        nxt_r = self.image_loader(self.path_list[3][index : index + self.seq_len])
        
        if self.transform:
            cur_l = self.transform(cur_l)
            cur_r = self.transform(cur_r)
            nxt_l = self.transform(nxt_l)
            nxt_r = self.transform(nxt_r)
        
        return self.convert_to_items(cur_l, cur_r, nxt_l, nxt_r, self.get_poses(index))
    
    def read_poses(self, path):
        poses = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pose = np.fromstring(line, dtype=float, sep=' ')
                pose = pose.reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses.append(pose)
        return poses

    def get_poses(self, index):
        poses = []
        for i in range(self.seq_len):
            poses.append(self.get_pose_difs(self.pose_list[index + i], self.pose_list[index + i + 1])) 
        return poses

    def get_pose_difs(self, prev_pose, new_pose):
        return new_pose - prev_pose

    def imread(self, path, flag=1):
        im = cv2.imread(path, flag)
        im = im.astype(np.float32) / 255.0
        return im

    def convert_to_items(self, cur_l, cur_r, nxt_l, nxt_r, pose):
        items = []
        for i in range(self.seq_len):
            items.append([cur_l[i], cur_r[i], nxt_l[i], nxt_r[i], pose[i]])
        return items

    