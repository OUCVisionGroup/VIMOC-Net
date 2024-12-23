import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
from torchvision import datasets, transforms

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,transform2=None,  train=False, seen=0, batch_size=1, num_workers=4):
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.transform2 = transform2
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

                        
        img_path = self.lines[index]
        
        prev_img,prev_png,img,png,prev_target,target,image1,image2,index = load_data(img_path, self.train)
        
        if self.transform is not None:
            prev_img = self.transform(prev_img)
            prev_png = self.transform2(prev_png)
            img = self.transform(img)
            png = self.transform2(png)

        return prev_img,prev_png,img,png,prev_target,target,image1,image2
