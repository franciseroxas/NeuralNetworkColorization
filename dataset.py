import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import os

class BW2ClrImageDataset(Dataset):
    def __init__(self, bw_img_dir, clr_img_dir):
        self.bw_img_dir = bw_img_dir
        self.clr_img_dir = clr_img_dir

    def __len__(self):
        return len(os.listdir(self.clr_img_dir)) #Names are shared between directories

    def __getitem__(self, idx):
        bw_img_path = self.bw_img_dir + os.listdir(self.clr_img_dir)[idx].replace('_clr.png', '_bw.png')
        clr_img_path = self.clr_img_dir + os.listdir(self.clr_img_dir)[idx]
        
        #Load in BGR 
        bwImage = torch.Tensor(cv2.imread(bw_img_path, cv2.IMREAD_GRAYSCALE)) / 255
        clrImage = torch.Tensor(cv2.imread(clr_img_path)) / 255
        clrImage = clrImage.unsqueeze(0)
        clrImage = torch.transpose(clrImage, 0, 3)

        return bwImage.unsqueeze(0), clrImage.squeeze() #feature dim is on the first dimension after batch size