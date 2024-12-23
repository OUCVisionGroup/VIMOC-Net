import h5py
import json
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model_DAFG import DAFG
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import time
import argparse
from torchvision import transforms

from sklearn.metrics import mean_squared_error,mean_absolute_error

import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'
parser = argparse.ArgumentParser(description='PyTorch DAFG')
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
args = parser.parse_args()

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the json file contains path of test images
test_json_path = '/data/macz/People-Flows-main/data/yu3/test3.json'


with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)



model = DAFG()
model = model.cuda()

raft = torch.nn.DataParallel(RAFT(args))
raft.load_state_dict(torch.load(args.model))

raft = raft.module
raft.to(DEVICE)
raft.eval()

# modify the path of saved checkpoint if necessary
checkpoint = torch.load('/data/macz/People-Flows-main/3/model3_test3.tar' )

model.load_state_dict(checkpoint['state_dict'])
model.eval()

pred= []
pred1= []
pred2= []
gt = []
p= []
g = []
total_processing_time = 0  
total_fps = 0 

def load_image(imfile):
    img = Image.open(imfile).resize((640, 360))  
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

for i in range(len(img_paths)):
    img_path = img_paths[i]

    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])
    if index > 0 and index < 501 :

        base = ((index - 1) // 10) * 10 + 1
        prev_index = int(max(base,index-1))

        prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))

        image1 = load_image(prev_img_path)
        image2 = load_image(img_path)

        
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        
        prev_img = Image.open(prev_img_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')

        prev_img = prev_img.resize((640,360))
        img = img.resize((640,360))

        prev_img = transform(prev_img).cuda()
        img = transform(img).cuda()


        gt_path = img_path.replace('.jpg','_resize.h5')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])

        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        img = img.unsqueeze(0)
        prev_img = prev_img.unsqueeze(0)
        
        flow_low, flow_up = raft(image1, image2, iters=20, test_mode=True)
        start_time = time.perf_counter()
        prev_flow,_ = model(prev_img,img,flow_up)
        
        
        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0,:] = 1.0
        mask_boundry[-1,:] = 1.0
        mask_boundry[:,0] = 1.0
        mask_boundry[:,-1] = 1.0
        
        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry
        
        
        prev_flow_inverse,_ = model(img,prev_img,flow_up)
        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry
        end_time = time.perf_counter()
        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).data.cpu().numpy()
 
        
        processing_time = end_time - start_time
        fps = 1 / processing_time
     
        total_processing_time += processing_time
        total_fps += fps

        target = target
        reconstruction_from_prev = (reconstruction_from_prev).data.cpu().numpy()
        reconstruction_from_prev_inverse = (reconstruction_from_prev_inverse).data.cpu().numpy()
        
        pred_sum = overall.sum()
        pred_sum1 = reconstruction_from_prev.sum()
        pred_sum2 = reconstruction_from_prev_inverse.sum()
        #print('pred:', pred_sum)
        pred.append(pred_sum)
        pred1.append(pred_sum1)
        pred2.append(pred_sum2)
        gt.append(np.sum(target))

average_fps = total_fps / 150        
print(f"FPS: {average_fps:.2f}")
mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))
mae1 = mean_absolute_error(pred1,gt)
rmse1 = np.sqrt(mean_squared_error(pred1,gt))
mae2 = mean_absolute_error(pred2,gt)
rmse2 = np.sqrt(mean_squared_error(pred2,gt))


min_mae = min(mae, mae1, mae2)
min_rmse = min(rmse, rmse1, rmse2)

print('pred:', pred) 
print('gt:', gt) 

print ('MAE: ',mae)
print ('RMSE: ',rmse)
print ('MAE: ',min_mae)
print ('RMSE: ',min_rmse)
