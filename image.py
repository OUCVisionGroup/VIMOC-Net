import os
from PIL import Image,ImageChops
import numpy as np
import h5py
import cv2
import torch
DEVICE = 'cuda'
def load_image(imfile):
    img = Image.open(imfile).resize((640, 360))  
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_data(img_path,train = True):
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])

    base = ((index - 1) // 16) * 16 + 1
    prev_index = int(max(base,index-1))
    post_index = int(max(base,index-1))

    prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))
    prev_png_path = os.path.join(img_folder,'%03d.png'%(prev_index))
    png_path = os.path.join(img_folder,'%03d.png'%(index))
    #flow_path = os.path.join(img_folder,'%03d.jpeg'%(index)) 
    post_img_path = os.path.join(img_folder,'%03d.jpg'%(post_index))
    
    image1 = load_image(prev_img_path)
    image2 = load_image(img_path)

    prev_gt_path = prev_img_path.replace('.jpg','_resize.h5')
    gt_path = img_path.replace('.jpg','_resize.h5')
    post_gt_path = post_img_path.replace('.jpg','_resize.h5')


    prev_img = Image.open(prev_img_path).convert('RGB')
    prev_png = Image.open(prev_png_path).convert('RGB')
    png = Image.open(png_path).convert('RGB')
    #flow = Image.open(flow_path).convert('L')
    img = Image.open(img_path).convert('RGB')
    post_img = Image.open(post_img_path).convert('RGB')

    #prev_img_L = Image.open(prev_img_path).convert('L')
    #img_L = Image.open(img_path).convert('L')
    #prev_img_L = prev_img_L.resize((640, 360), Image.ANTIALIAS)
    #img_L = img_L.resize((640, 360), Image.ANTIALIAS)
    #prev_img_np = np.array(prev_img_L)
    #img_np = np.array(img_L)

    #flow = cv2.calcOpticalFlowFarneback(prev_img_np, img_np, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #flow = np.transpose(flow, (2, 0, 1))

    prev_img = prev_img.resize((640,360))
    prev_png = prev_png.resize((80,45))
    img = img.resize((640,360))

    png = png.resize((80,45))
    post_img = post_img.resize((640,360))

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    gt_file.close()
    
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    prev_gt_file = h5py.File(prev_gt_path)
    prev_target = np.asarray(prev_gt_file['density'])
    prev_gt_file.close()
    prev_target = cv2.resize(prev_target,(int(prev_target.shape[1]/8),int(prev_target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    post_gt_file = h5py.File(post_gt_path)
    post_target = np.asarray(post_gt_file['density'])
    post_gt_file.close()
    post_target = cv2.resize(post_target,(int(post_target.shape[1]/8),int(post_target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    
    return prev_img,prev_png,img,png,prev_target,target,image1,image2,index

