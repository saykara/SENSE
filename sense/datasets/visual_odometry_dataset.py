import torch.utils.data as data
import os
import cv2
import numpy as np

class VODataset(data.Dataset):
    def __init__(self, data, transform, seq_len = 5):
        super(VODataset, self).__init__()
        self.data = data
        self.image_loader = self.imread
        self.transform = transform
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        result = []
        for i in range(self.seq_len):
            cur_l = self.image_loader(self.data[index][i][0])
            cur_r = self.image_loader(self.data[index][i][1])
            nxt_l = self.image_loader(self.data[index][i][2])
            nxt_r = self.image_loader(self.data[index][i][3])
            pose = self.data[index][i][4]
            
            if self.transform:
                cur_l = self.transform(cur_l)
                cur_r = self.transform(cur_r)
                nxt_l = self.transform(nxt_l)
                nxt_r = self.transform(nxt_r)
            result.append([cur_l, cur_r, nxt_l, nxt_r, pose])
        result.append(self.data[index][self.seq_len])
        return result
    
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

    