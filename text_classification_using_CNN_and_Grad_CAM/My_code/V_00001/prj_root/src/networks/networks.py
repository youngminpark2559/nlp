# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/networks/
# rm e.l && python networks.py 2>&1 | tee -a e.l && code e.l

# ================================================================================
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os
import copy
import glob
import cv2
import natsort 
from PIL import Image
from skimage.transform import resize
import scipy.misc
from sklearn import svm

# ================================================================================
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets,models,transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

# ================================================================================
from src.networks import cbam as cbam

# ================================================================================
def init_weights(m):
  if type(m) == nn.Linear or\
     type(m) == nn.Conv2d or\
     type(m) == nn.ConvTranspose2d:
    torch.nn.init.xavier_uniform_(m.weight.data)
    # torch.nn.init.xavier_uniform_(m.bias)
    # m.bias.data.fill_(0.01)

# ================================================================================
def crop_and_concat(upsampled,bypass, crop=False):
  if crop:
    c=(bypass.size()[2]-upsampled.size()[2])//2
    bypass=F.pad(bypass,(-c,-c,-c,-c))
  return torch.cat((upsampled,bypass),1)

# ================================================================================
class Interpolate(nn.Module):
  def __init__(
    self,size=None,scale_factor=None,mode="bilinear",align_corners=True):
    super(Interpolate,self).__init__()
    self.interp=F.interpolate
    self.size=size
    self.scale_factor=scale_factor
    self.mode=mode
    self.align_corners=align_corners

  def forward(self,x):
    x=self.interp(
      x,size=self.size,scale_factor=self.scale_factor,
      mode=self.mode,align_corners=self.align_corners)
    return x

# ================================================================================
class CNN_for_text_classification(nn.Module):
    
    def __init__(self,args):
        super(CNN_for_text_classification, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        print("V",V)
        print("D",D)
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(float(args.dropout))
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # print("x",x.shape)
        # torch.Size([2, 200])

        x = self.embed(x)  # (N, W, D)
        # print("x",x.shape)
        # torch.Size([2, 200, 128])
        
        # ================================================================================
        # print("self.args.static",self.args.static)
        # False
        # if self.args.static:
        #     x = Variable(x)
        
        # ================================================================================
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print("x",x.shape)
        # torch.Size([2, 1, 200, 128])

        # ================================================================================
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x_conv1=self.convs1[0](x)
        x_conv1=F.relu(x_conv1)
        x_conv1=x_conv1.squeeze(3)
        # print("x_conv1",x_conv1.shape)
        # torch.Size([2, 100, 198])

        x_conv2=self.convs1[1](x)
        x_conv2=F.relu(x_conv2)
        x_conv2=x_conv2.squeeze(3)
        # print("x_conv2",x_conv2.shape)
        # torch.Size([2, 100, 197])

        x_conv3=self.convs1[2](x)
        x_conv3=F.relu(x_conv3)
        x_conv3=x_conv3.squeeze(3)
        # print("x_conv3",x_conv3.shape)
        # torch.Size([2, 100, 196])

        x=[x_conv1,x_conv2,x_conv3]

        # ================================================================================
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x0_size_dim2=x[0].size(2)
        # print("x[0]",x[0].shape)
        # torch.Size([2, 100, 198])
        x0_pool=F.max_pool1d(x[0],x0_size_dim2).squeeze(2)
        # print("x0_pool",x0_pool.shape)
        # torch.Size([2, 100])

        x1_size_dim2=x[1].size(2)
        # print("x[1]",x[1].shape)
        # torch.Size([2, 100, 197])
        x1_pool=F.max_pool1d(x[1],x1_size_dim2).squeeze(2)
        # print("x1_pool",x1_pool.shape)
        # torch.Size([2, 100])

        x2_size_dim2=x[2].size(2)
        # print("x[2]",x[2].shape)
        # torch.Size([2, 100, 196])
        x2_pool=F.max_pool1d(x[2],x2_size_dim2).squeeze(2)
        # print("x2_pool",x2_pool.shape)
        # torch.Size([2, 100])

        x=[x0_pool,x1_pool,x2_pool]
        # print("x",x.shape)

        # ================================================================================
        x = torch.cat(x, 1)
        # print("x",x.shape)
        # torch.Size([2, 300])

        # ================================================================================
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # print("x",x.shape)
        # torch.Size([2, 300])

        # ================================================================================
        logit = self.fc1(x)  # (N, C)
        # print("logit",logit.shape)
        # torch.Size([2, 2])

        # print("logit",logit)
        # tensor([[ 0.3338, -0.4838],
        #         [-0.1408, -0.3446]], device='cuda:0', grad_fn=<AddmmBackward>)
        
        return logit
