import torch.utils.data as data
import os
import cv2
import numpy as np
import torch
import threading
import torch.nn as nn 

from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.lib.nn import DataParallelWithCallback
from sense.models.dummy_scene import SceneNet, SceneNeXt

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
    if torch.min(item) < e_mi:
        e_mi = torch.min(item)
        print("Eva min: ", e_mi)
    if torch.max(item) > e_ma:
        e_ma = torch.max(item)
        print("Eva max: ",e_ma)

class PreprocessingCollateFn(object):
    def __init__(self, optical_flow_model_path, encoder_path, flow_transform, final_transform, args):
        self.flow_transform = flow_transform
        self.final_transform = final_transform
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ego_model = DataParallelWithCallback(EGOAutoEncoder(args)).cuda()
        ckpt = torch.load(encoder_path)
        state_dict = ckpt['state_dict']
        ego_model.load_state_dict(state_dict)
        self.encoder = ego_model.module.encoder
        self.encoder.to(device)
        self.encoder.eval()
        
        self.optical_flow_model = DataParallelWithCallback(SceneNeXt(args)).cuda()
        ckpt = torch.load(optical_flow_model_path)
        self.optical_flow_model.load_state_dict(ckpt['state_dict'])
        self.optical_flow_model.to(device)
        self.optical_flow_model.eval()

        self.maxpool = nn.MaxPool2d(2)
        
    def __call__(self, x):
        batch, pose = x
        batch, pose = batch.to("cuda"), pose.to("cuda")
        flows = []
        with torch.no_grad():
            for item in batch:
                new_dim = item.size(1) * item.size(2)
                # Reshape the tensor to the desired shape
                output_tensor1 = item.view(item.size(0), new_dim, item.size(3), item.size(4))[:, :3, :, :]
                output_tensor2 = item.view(item.size(0), new_dim, item.size(3), item.size(4))[:, 3:, :, :]
                flow = self.optical_flow_model(output_tensor1, output_tensor2)
                # flow = self.transform_flow(flow)
                flow = self.encoder(flow)
                min_max_eva(flow)
                flow = self.transform_final(flow)
                flow = self.maxpool(flow)
                flows.append(flow)
        return torch.stack(flows), pose
    
    def transform_flow(self, flow):
        return self.flow_transform(flow)

    def transform_final(self, flow):
        return self.final_transform(flow[4])
    
class VODataset(data.Dataset):
    def __init__(self, data, input_transform, seq_len = 5):
        super(VODataset, self).__init__()
        self.data = data
        self.image_loader = self.imread
        self.input_transform = input_transform
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = []
        for i in range(self.seq_len):
            cur_l = self.image_loader(self.data[index][i][0])
            nxt_l = self.image_loader(self.data[index][i][1])
            
            if self.input_transform:
                cur_l = self.input_transform(cur_l)
                nxt_l = self.input_transform(nxt_l)
            
            seq.append(torch.stack([cur_l, nxt_l]))
        pose = torch.tensor(self.data[index][self.seq_len])
        pose = pose.to(torch.float32)
        return torch.stack(seq), pose
    
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

    