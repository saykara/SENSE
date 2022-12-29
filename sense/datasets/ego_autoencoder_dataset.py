import torch.utils.data as data
import os.path
import cv2
import numpy as np

import sense.datasets.dataset_utils as du

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
        
        if self.transform:
            flo = self.transform(flo)
        #image = np.einsum('ijk->kji', image)
        return flo

def imread(im_path, flag=1):
    im = cv2.imread(im_path, flag)
    im = im.astype(np.float32) / 255.0
    return im

class EGOAutoencoderImageDataset(data.Dataset):
    def __init__(self, root, path_list, transform):
        super(EGOAutoencoderImageDataset, self).__init__()
        self.root = root
        self.path_list = path_list
        self.loader = imread
        self.transform = transform
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        cur_l = self.loader(self.path_list[index][0])
        nxt_l = self.loader(self.path_list[index][1])
        if self.transform:
            cur_l = self.transform(cur_l)
            nxt_l = self.transform(nxt_l)
        #image = np.einsum('ijk->kji', image)
        return cur_l, nxt_l
    
    