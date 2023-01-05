import torch.utils.data as data
import os
import cv2
import numpy as np
import torch
import threading

from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.lib.nn import DataParallelWithCallback
from sense.models.dummy_scene import SceneNet

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

class PreprocessingCollateFn(object):
    def __init__(self, optical_flow_model_path, encoder_path, flow_transform, final_transform, args):
        self.flow_transform = flow_transform
        self.final_transform = final_transform
        
        ego_model = DataParallelWithCallback(EGOAutoEncoder(args))
        ckpt = torch.load(encoder_path)
        state_dict = ckpt['state_dict']
        ego_model.load_state_dict(state_dict)
        self.encoder = ego_model.module.encoder
        self.encoder.eval()
        
        self.optical_flow_model = DataParallelWithCallback(SceneNet(args))
        ckpt = torch.load(optical_flow_model_path)
        self.optical_flow_model.load_state_dict(ckpt['state_dict'])
        self.optical_flow_model.eval()
        
        self.flow_lock = threading.Lock()
        self.enc_lock = threading.Lock()

    def __call__(self, batch):
        flows = []
        poses = []
        for sample in batch:
            seq = []
            for img in sample[0]:
                with torch.no_grad():
                    self.flow_lock.acquire()
                    flow = self.optical_flow_model(img[0], img[1])
                    self.flow_lock.release()
                    flow = self. transform_flow(flow)   
                    self.enc_lock.acquire()
                    flow = self.encoder(flow)
                    self.enc_lock.release()
                    flow = self.transform_final(flow)
                    seq.append(flow)
            flows.append(torch.stack(seq))
            poses.append(sample[1])
        return torch.stack(flows), torch.stack(poses)
    
    def transform_flow(self, flow):
        flow = self.flow_transform(flow)
        return torch.unsqueeze(flow, 0)

    def transform_final(self, flow):
        flow = flow[4]
        flow = torch.squeeze(flow, 0)
        return self.final_transform(flow)
    
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
            
            seq.append([cur_l, nxt_l])
        pose = torch.tensor(self.data[index][self.seq_len])
        pose = pose.to(torch.float32)
        return seq, pose
    
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

    