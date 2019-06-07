
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils_for_dataset/dataset_cgmit.py
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/unit_test/Test_dataset_cgmit.py

# ================================================================================
import csv
import numpy as np
import pandas as pd
from random import shuffle

# ================================================================================
import torch.utils.data as data

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

# ================================================================================
class Custom_DS(data.Dataset):
  def __init__(self,text_data_path,args):
    contents_txt_data=pd.read_csv(text_data_path,encoding='utf8')
    # print("contents_txt_data",contents_txt_data)
    # 0   Nausea, diarrhea, shakiness, fatigue and feeling cold all experienced from this drug.      0
    # 1   Well besides the constant diarrhea, everything else is good.                               1
    # 2   Taking 500 mg per day. Some stomach discomfort right away and brief diarrhea. In           0
    # 3   I am thankful that Metformin has kept me from having Type 2 Diabetes however the           1

    text_data=contents_txt_data.iloc[:,0].tolist()
    text_label=contents_txt_data.iloc[:,1].tolist()
    # print("text_data",text_data)
    # print("text_label",text_label)
    
    zipped=np.array(list(zip(text_data,text_label)))
    # print("zipped",zipped)
    # [['Nausea, diarrhea, shakiness, fatigue and feeling cold all experienced from this drug.' '0']
    #  ['Well besides the constant diarrhea, everything else is good.' '1']

    # ================================================================================
    # shuffle(zipped)

    # ================================================================================
    self.trn_pairs=zipped

    # ================================================================================
    # instance of argument 
    self.args=args

    # ================================================================================
    self.nb_trn_imgs=len(zipped)
    # print("self.nb_trn_imgs",self.nb_trn_imgs)
    # 16

  # ================================================================================
  def __len__(self):
    return self.nb_trn_imgs

  # ================================================================================
  def __getitem__(self,idx):
    one_pair=self.trn_pairs[idx]
    return one_pair

# ================================================================================
class Custom_DS_vali(data.Dataset):
  def __init__(self,single_vali_k,single_vali_lbl_k,args):
    # print("single_vali_k",single_vali_k)
    # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png'

    # print("single_vali_lbl_k",single_vali_lbl_k)
    # [['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0' '16 0']
    #  ['001838f8-bbca-11e8-b2bc-ac1f6b6435d0' '18']

    # print("single_vali_lbl_k",single_vali_lbl_k[:,1])
    # ['7 1 2 0' '5' '1' ... '5 0' '19 23' '18']

    zipped=np.array(list(zip(single_vali_k,single_vali_lbl_k[:,1])))

    # ================================================================================
    zipped_new=[[list(one_protein_pair[0]),one_protein_pair[1]] for one_protein_pair in zipped]
    # print("zipped_new",zipped_new)
    # [[['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_blue.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_green.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_red.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_yellow.png'],
    #  '16 0'],

    self.vali_pairs=zipped_new
    # print("self.vali_pairs",self.vali_pairs)

    # ================================================================================
    # instance of argument 
    self.args=args

    # ================================================================================
    self.nb_vali_imgs=len(single_vali_k)
    # print("self.nb_vali_imgs",self.nb_vali_imgs)
    # 10358

  # ================================================================================
  def __len__(self):
    return self.nb_vali_imgs

  # ================================================================================
  def __getitem__(self,idx):
    one_pair=self.vali_pairs[idx]
    return one_pair
