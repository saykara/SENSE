
import torch
import torch.utils.data as data
import os
import os.path
import cv2
import numpy as np
import pdb
import pickle
from PIL import Image

import sense.datasets.dataset_utils as du

def imreader(path, flag=0):
    im = cv2.imread(path, flag)
    im = im.astype(np.float32) / 255.0
    return im

# https://stackoverflow.com/a/51272988
def pad(img):
    """Return bottom right padding."""
    h_pad = 32 - img.shape[0] % 32
    w_pad = 32 - img.shape[1] % 32
    zeros = np.zeros((img.shape[0] + h_pad, img.shape[1] + w_pad))
    zeros[:img.shape[0], :img.shape[1]] = img
    zeros = zeros.astype(np.float32) / 255.0
    return zeros

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
        image = pad(self.loader(self.path_list[index]))
        if self.transform:
            # Array to PIL - PIL to array, readdress this
            image = Image.fromarray(image)
            image = self.transform(image)
            image = np.asarray(image)
        return image
    
    