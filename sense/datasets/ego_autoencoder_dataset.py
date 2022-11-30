
import torch
import torch.utils.data as data
import os
import os.path
import cv2
import numpy as np
import pdb
import pickle
from PIL import Image


def imreader(root, path, flag=1):
    im = np.array(Image.open(os.path.join(root, path)))
    #im = cv2.imread(os.path.join(root, path), flag)
    im = im.astype(np.float32) / 255.0
    return im
    
class EGOFlowDataset(data.Dataset):
    def __init__(self, root, path_list, transform):
        super(EGOFlowDataset, self).__init__()
        self.root = root
        self.path_list = path_list
        self.loader = imreader
        self.transform = transform
        
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        image = self.loader(self.root, self.path_list[index])
        if self.transform:
            image = self.transform(image)
        return image
    
    