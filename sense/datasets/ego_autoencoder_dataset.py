import torch.utils.data as data
import os.path
import cv2
import numpy as np
import torch

import sense.datasets.dataset_utils as du

f_mi = -5.0
f_ma = 5.0
        
        
def load_flow(path):
    if path.endswith('.pfm'):
        flow, _ = du.load_pfm(path)
        return flow[:, :, :2]
    elif path.endswith('.flo'):
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        return np.resize(data, (w, h, 2))
    else:
        ext = path.split(".")[-1]
        assert f"Invalid flow file extension ({ext})."

class EGOFlowDataset(data.Dataset):
    def __init__(self, root, path_list, transform):
        super(EGOFlowDataset, self).__init__()
        self.root = root
        self.path_list = path_list
        self.loader = load_flow
        self.transform = transform
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        flo = self.loader(self.path_list[index])
#        self.min_max_flow(flo, index)
        if self.transform:
            flo = self.transform(flo)
        #image = np.einsum('ijk->kji', image)
        return flo
    
#    def min_max_flow(self, item, index):
#        global f_mi, f_ma
#        if np.min(item) < f_mi:
#            f_mi = np.min(item)
#            print("Flo min: ",f_mi)
#        if np.max(item) > f_ma:
#            f_ma = np.max(item)
#            print("Flo max: ",f_ma)
#        
#        if np.min(item) < -800. or np.max(item) > 800.:
#            print(np.min(item), " - ",  np.max(item), " - ", self.path_list[index])

def imread(im_path, flag=1):
    im = cv2.imread(im_path, flag)
    im = im.astype(np.float32) / 255.0
    return im

class EGOAutoencoderImageDataset(data.Dataset):
    def __init__(self, root, path_list, transform, flow_transform, model):
        super(EGOAutoencoderImageDataset, self).__init__()
        self.root = root
        self.path_list = path_list
        self.loader = imread
        self.transform = transform
        self.model = model
        self.flow_transform = flow_transform
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        cur_l = self.loader(self.path_list[index][0])
        nxt_l = self.loader(self.path_list[index][1])        
        if self.transform:
            cur_l = self.transform(cur_l)
            nxt_l = self.transform(nxt_l)
        #image = np.einsum('ijk->kji', image)
        with torch.no_grad():
            flow = self.model(cur_l, nxt_l)

        if self.flow_transform:
            flow = self.flow_transform(flow)
        
        return flow
    
    