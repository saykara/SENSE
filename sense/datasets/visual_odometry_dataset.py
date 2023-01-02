import torch.utils.data as data
import os
import cv2
import numpy as np
import torch
from tools.demo import image_to_tensor, run_holistic_scene_model

f_mi = -5.0
f_ma = 5.0
e_mi = -5.0
e_ma = 5.0

def min_max_flow(item):
    global f_mi, f_ma
    if item < f_mi:
        f_mi = item
        print("Flo min: ",f_mi)
    if item > f_ma:
        f_ma = item
        print("Flo max: ",f_ma)
        
def min_max_eva(item):
    global e_mi, e_ma
    if item < e_mi:
        e_mi = item
        print("Eva min: ", e_mi)
    if item > e_ma:
        e_ma = item
        print("Eva max: ",e_ma)
    
class VODataset(data.Dataset):
    def __init__(self, data, input_transform, flow_transform, final_transform, optical_flow_model, encoder, seq_len = 5):
        super(VODataset, self).__init__()
        self.data = data
        self.image_loader = self.imread
        self.input_transform = input_transform
        self.flow_transform = flow_transform
        self.final_transform = final_transform
        self.seq_len = seq_len
        self.optical_flow_model = optical_flow_model
        self.encoder = encoder
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        flows = []
        for i in range(self.seq_len):
            cur_l = self.image_loader(self.data[index][i][0])
            nxt_l = self.image_loader(self.data[index][i][1])
            
            if self.input_transform:
                cur_l = self.input_transform(cur_l)
                nxt_l = self.input_transform(nxt_l)
            with torch.no_grad():
                flow = self.optical_flow_model(cur_l, nxt_l)
            #min_max_flow(torch.max(flow).item())
            #min_max_flow(torch.min(flow).item())

            if self.flow_transform:
                flow = self.flow_transform(flow)
                flow = torch.unsqueeze(flow, 0)
            with torch.no_grad():    
                flow = self.encoder(flow)
                flow = flow[4]
                flow = torch.squeeze(flow, 0)
            #min_max_eva(torch.max(flow).item())
            #min_max_eva(torch.min(flow).item())
            if self.final_transform:
                flow = self.final_transform(flow)
            
            flows.append(flow)
        flows = torch.stack(flows)
        pose = torch.tensor(self.data[index][self.seq_len])
        pose = pose.to(torch.float32)
        return flows, pose, self.data[index][self.seq_len + 1]
    
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

    