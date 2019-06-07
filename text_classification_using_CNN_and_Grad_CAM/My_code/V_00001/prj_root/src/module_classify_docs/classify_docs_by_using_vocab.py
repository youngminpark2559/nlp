import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys,os,copy,argparse
import time,timeit,datetime
import glob,natsort
import cv2
from PIL import Image
from skimage.transform import resize
from skimage.restoration import (denoise_tv_chambolle,denoise_bilateral,
                                 denoise_wavelet,estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,\
                            precision_score,recall_score,fbeta_score,f1_score,roc_curve
import scipy.misc
import scipy.optimize as opt
import scipy.special
from sklearn import svm
from skimage.viewer import ImageViewer
from random import shuffle
import gc
import Augmentor
import traceback
import subprocess
import csv
from xgboost import XGBClassifier

# ================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.autograd import gradcheck
from torch.optim.lr_scheduler import StepLR

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_net as utils_net
from src.utils import utils_pytorch as utils_pytorch
from src.utils import utils_data as utils_data
from src.utils import utils_hook_functions as utils_hook_functions
from src.utils import utils_visualize_gradients as utils_visualize_gradients

from src.utils_for_train import train_over_sentiment_dataset as train_over_sentiment_dataset

from src.utils_NLP import utils_find_similarity_of_docs_against_vocab_files as utils_find_similarity_of_docs_against_vocab_files

from src.utils_for_dataset import custom_ds as custom_ds
from src.utils_for_dataset import custom_ds_test as custom_ds_test

from src.loss_functions import loss_functions_module as loss_functions_module
from src.metrics import metrics_module as metrics_module

from src.api_model import model_api_module as model_api_module
from src.api_text_file_path import text_file_path_api_module as text_file_path_api_module

from src.utils_analyzing_result import grad_cam as grad_cam

# ================================================================================
def train(args):
  # dir_Docs="/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Docs/*.txt"
  # dir_Vocab="/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Vocab/*.txt"

  # utils_find_similarity_of_docs_against_vocab_files.calculate(dir_Docs,dir_Vocab,args)

  # utils_find_similarity_of_docs_against_vocab_files.color_word(dir_Docs,dir_Vocab,args)
  # afaf

  vocab_txt_path="/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00002_Grad_CAM_on_text_classification/My_code/Data/vocab.txt"
  text_data_path="/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00002_Grad_CAM_on_text_classification/My_code/Data/text_data.csv"

  # ================================================================================
  contents,num_line=utils_common.return_path_list_from_txt(vocab_txt_path)
  contents=list(map(lambda x:x.replace("\n",""),contents))
  # print("contents",contents)
  # ['<unk>', '<pad>', '', 'the', ',', 'a', 'and', 'of', 'to', 'is', 'in', 'that', 'it', 'as', 'but', 'with', 'film', 'this', 'for', 'its', 'an', 'movie', "it 's", 'be', 'on', 'you', 'not', 'by', 
  # print("contents",len(contents))
  # 21114

  # ================================================================================
  args.__setattr__("embed_num",len(contents))
  args.__setattr__("embed_dim",128)
  args.__setattr__("class_num",2)
  args.__setattr__("kernel_sizes",[3,4,5])
  args.__setattr__("kernel_num",100)
  args.__setattr__("save_dir",os.path.join(".",datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
  
  # ================================================================================
  model_API_obj=model_api_module.Model_API_class(args)
  # print("model_API_obj",model_API_obj)
  # <src.api_model.model_api_module.Model_API_class object at 0x7f911af64518>
  # afaf 2: model_API_obj=model_api_module.Model_API_class(args)

  # ================================================================================
  epoch=int(args.epoch)
  batch_size=int(args.batch_size)
  # print("epoch",epoch)
  # print("batch_size",batch_size)
  # 9
  # 2
  
  # ================================================================================
  # c loss_list: list which will stores loss values to plot loss
  loss_list=[]
  f1_score_list=[]

  # ================================================================================
  for one_ep in range(epoch): # @ Iterates all epochs
    custom_ds_obj=custom_ds.Custom_DS(text_data_path,args)

    custom_ds_iter=iter(custom_ds_obj)

    num_all_sentences=len(custom_ds_obj)
    # print("num_all_sentences",num_all_sentences)
    # 16

    args.__setattr__("num_all_sentences",num_all_sentences)

    num_iteration_for_iter=int(int(num_all_sentences)/int(args.batch_size))
    # print("num_iteration_for_iter",num_iteration_for_iter)
    # 8

    # ================================================================================
    for one_iter in range(num_iteration_for_iter):
      # @ Remove gradients
      model_API_obj.remove_existing_gradients_before_starting_new_training()

      # ================================================================================
      train_over_sentiment_dataset.train(custom_ds_iter,batch_size,model_API_obj,contents,args)
