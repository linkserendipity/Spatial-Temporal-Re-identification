# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision import datasets
import os
import scipy.io
import math
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir',default="/home/ccc/Link/data/dataset/market_rename/",type=str, help='./train_data')
parser.add_argument('--name', default='ft_ResNet50_pcb_market_e', type=str, help='save model path')

opt = parser.parse_args()
name = opt.name
data_dir = opt.data_dir
model_path = '../ST_model'

def get_id(img_path):
    camera_id = []
    labels = []
    frames = []
    for path, v in img_path:                    # * path='/home/ccc/Link/data/dataset/market_rename/train/0002/0002_c1_f0000451_03.jpg'
        filename = path.split('/')[-1]          # * filename='0002_c1_f0000451_03.jpg'
        label = filename[0:4]                   # * label=0002 person id
        camera = filename.split('c')[1]         # * camera = '1_f0000451_03.jpg'  camera[0]='1'
        # frame = filename[9:16]
        frame = filename.split('_')[2][1:]      # * frame='0000451'
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        frames.append(int(frame))
    return camera_id, labels, frames           

def spatial_temporal_distribution(camera_id, labels, frames):
    class_num=751 #* 751 people
    max_hist = 5000 #! max_hist??
    spatial_temporal_sum = np.zeros((class_num,8))                        #!!! 8 cameras???
    spatial_temporal_count = np.zeros((class_num,8))
    eps = 0.0000001
    interval = 100.0
    
    for i in range(len(camera_id)):         #! camera_id is a list of 751 person?
        label_k = labels[i]                 #### not in order, done
        cam_k = camera_id[i]-1              ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        spatial_temporal_sum[label_k][cam_k]=spatial_temporal_sum[label_k][cam_k]+frame_k
        spatial_temporal_count[label_k][cam_k] = spatial_temporal_count[label_k][cam_k] + 1
    spatial_temporal_avg = spatial_temporal_sum/(spatial_temporal_count+eps)          # spatial_temporal_avg: 751 ids, 8cameras, center point
    
    distribution = np.zeros((8,8,max_hist))
    for i in range(class_num):
        for j in range(8-1):
            for k in range(j+1,8):
                if spatial_temporal_count[i][j]==0 or spatial_temporal_count[i][k]==0:
                    continue 
                st_ij = spatial_temporal_avg[i][j]
                st_ik = spatial_temporal_avg[i][k]
                if st_ij>st_ik:
                    diff = st_ij-st_ik
                    hist_ = int(diff/interval)
                    distribution[j][k][hist_] = distribution[j][k][hist_]+1     # [big][small]
                else:
                    diff = st_ik-st_ij
                    hist_ = int(diff/interval)
                    distribution[k][j][hist_] = distribution[k][j][hist_]+1
    
    sum_ = np.sum(distribution,axis=2)
    for i in range(8):
        for j in range(8):
            distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)
    
    return distribution                    # [to][from], to xxx camera, from xxx camera

def gaussian_func(x, u, o=0.1):
    if (o == 0):
        print("In gaussian, o shouldn't equel to zero")
        return 0
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(math.pow(x - u, 2)) / (2 * math.pow(o, 2))
    return temp1 * math.exp(temp2)


def gauss_smooth(arr):
    # print(gaussian_func(0,0))
    for u, element in enumerate(arr):
        # print(u," ",element)
        if element != 0:
            for index in range(0, 3000):
                arr[index] = arr[index] + element * gaussian_func(index, u)

    sum = 0
    for v in arr:
        sum = sum + v
    if sum==0:
        return arr
    for i in range(0,3000):
        arr[i] = arr[i] / sum
    return arr


transform_train_list = [
        transforms.Resize(144, interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

#* image_data: x 是文件夹序号，person id 0002对应x=0，0007对应x=1
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,transform_train_list) for x in ['train_all']} #* transform_train_list有什么用?
train_path = image_datasets['train_all'].imgs   # ! len(train_path)=12936
train_cam, train_label, train_frames = get_id(train_path)
#! train_path[0]=('/home/ccc/Link/data/dataset/market_rename/train_all/0002/0002_c1_f0000451_03.jpg', 0) 

train_label_order = []
for i in range(len(train_path)):
    train_label_order.append(train_path[i][1]) 
# * train_path[i][1]=0,0,...46个0,1,1,1,1

#todo distribution = spatial_temporal_distribution(train_cam, train_label, train_frames) 
#! label=0002 0007 change to order=0,0...,1,1...1,2,2...2,3
distribution = spatial_temporal_distribution(train_cam, train_label_order, train_frames)

# for i in range(0,8):
#     for j in range(0,8):
#         print("gauss "+str(i)+"->"+str(j))
#         gauss_smooth(distribution[i][j])
#?????
result = {'distribution':distribution} 
scipy.io.savemat(model_path+'/'+name+'/'+'pytorch_result2.mat', result)
