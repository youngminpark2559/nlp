
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils_for_dataset/dataset_cgmit.py
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/unit_test/Test_dataset_cgmit.py

# ================================================================================
import csv
import numpy as np
import pandas as pd
from random import shuffle
import traceback

from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

# ================================================================================
import torch.utils.data as data
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

from src.utils_for_dataset import get_batch_size_paths_using_iter as get_batch_size_paths_using_iter

# ================================================================================
stop_words=set(stopwords.words('english'))
wnl=WordNetLemmatizer()
tokenizer=RegexpTokenizer(r'\w+')

def train(custom_ds_iter,batch_size,model_API_obj,contents,args):
  bs_paths_dense=get_batch_size_paths_using_iter.return_bs_paths(custom_ds_iter,batch_size)
  # print("bs_paths_dense",bs_paths_dense)
  # [array(['Nausea, diarrhea, shakiness, fatigue and feeling cold all experienced from this drug.','0'], dtype='<U577'), 
  #  array(['Well besides the constant diarrhea, everything else is good.','1'], dtype='<U577')]

  # ================================================================================
  # print("contents",contents)
  # ['<unk>', '<pad>', '', 'the', ',', 'a', 'and', 'of', 'to', 'is', 'in', 'that', 'it', 'as', 'but', 'with', 'film', 'this', 'for', 'its', 'an', 'movie', "it 's", 'be', 'on', 'you', 'not', 'by', 

  idx_word_dict={}
  word_idx_dict={}
  for i,one_word in enumerate(contents):
    idx_word_dict[i]=one_word
    word_idx_dict[one_word]=i

  # print("idx_word_dict",idx_word_dict)
  # {0: '<unk>', 1: '<pad>', 2: '', 3: 'the', 4: ',', 5: 'a', 6: 'and', 7: 'of', 8: 'to', 9: 'is', 10: 'in', 11: 'that', 12: 'it', 13: 'as', 14: 'but', 15: 'with', 16: 'film', 17: 'this', 18: 

  # print("word_idx_dict",word_idx_dict)
  # {'<unk>': 0, '<pad>': 1, '': 2, 'the': 3, ',': 4, 'a': 5, 'and': 6, 'of': 7, 'to': 8, 'is': 9, 'in': 10, 'that': 11, 'it': 12, 'as': 13, 'but': 14, 'with': 15, 'film': 16, 'this': 17, 'for': 

  # ================================================================================
  token_all_sentences=[]
  lbl_all_sentences=[]
  for one_sent in bs_paths_dense:
    sent=one_sent[0]
    lbl=one_sent[1]
    # print("sent",sent)
    # Nausea, diarrhea, shakiness, fatigue and feeling cold all experienced from this drug.
    
    # print("lbl",lbl)
    # 0

    # ================================================================================
    doc_str_wo_punctu=tokenizer.tokenize(sent)
    # print("doc_str_wo_punctu",doc_str_wo_punctu)
    # ['Nausea', 'diarrhea', 'shakiness', 'fatigue', 'and', 'feeling', 'cold', 'all', 'experienced', 'from', 'this', 'drug']

    doc_str_wo_punctu=" ".join(doc_str_wo_punctu)

    word_tokens=word_tokenize(doc_str_wo_punctu)
    # print("word_tokens",word_tokens)
    # ['Nausea', 'diarrhea', 'shakiness', 'fatigue', 'and', 'feeling', 'cold', 'all', 'experienced', 'from', 'this', 'drug']

    # ================================================================================
    result=[]
    for w in word_tokens: 
      if w not in stop_words:
        result.append(w)

    # print("result",result)
    # ['Nausea', 'diarrhea', 'shakiness', 'fatigue', 'feeling', 'cold', 'experienced', 'drug']

    # ================================================================================
    sent=" ".join(result)
    # print("sent",sent)
    # Nausea diarrhea shakiness fatigue feeling cold experienced drug

    sent_after_lemma=" ".join([wnl.lemmatize(i) for i in sent.split()])
    # print("sent_after_lemma",sent_after_lemma)
    # Nausea diarrhea shakiness fatigue feeling cold experienced drug

    # ================================================================================
    doc_str=" ".join(result)
    # print("doc_str",doc_str)
    # Nausea diarrhea shakiness fatigue feeling cold experienced drug

    doc_list=word_tokenize(doc_str)
    # print("doc_list",doc_list)
    # ['Nausea', 'diarrhea', 'shakiness', 'fatigue', 'feeling', 'cold', 'experienced', 'drug']

    # ================================================================================
    token_sentence=[]
    for one_unit_word in doc_list:
      # print("one_unit_word",one_unit_word)
      # Nausea

      try:
        idx_num=word_idx_dict[one_unit_word]
        # print("idx_num",idx_num)
        token_sentence.append(idx_num)
      except:
        # print(traceback.format_exc())
        idx_num=0
        token_sentence.append(idx_num)
        
    # print("token_sentence",token_sentence)
    # [0, 0, 0, 0, 352, 573, 3994, 2121]

    len_token_sentence=len(token_sentence)
    # print("len_token_sentence",len_token_sentence)
    # 8

    lack=200-len_token_sentence
    # print("lack",lack)
    # 192

    padding=lack*[1]
    # print("padding",padding)
    # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

    padded_tok=token_sentence+padding
    # print("padded_tok",padded_tok)
    # [0, 0, 0, 0, 352, 573, 3994, 2121, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

    # print(len(padded_tok))
    # 200

    token_all_sentences.append(padded_tok)
    lbl_all_sentences.append(lbl)

  # print("token_all_sentences",token_all_sentences)
  # [[0, 0, 0, 0, 352, 573, 3994, 2121, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

  # print("lbl_all_sentences",lbl_all_sentences)
  # ['0', '1']

  lbl_all_sentences=list(map(lambda x:int(x),lbl_all_sentences))

  token_all_sentences=Variable(torch.tensor(token_all_sentences).cuda())
  # print("token_all_sentences",token_all_sentences)

  label_all_sentences=Variable(torch.tensor(lbl_all_sentences).cuda())
  
  logit=model_API_obj.gen_net(token_all_sentences)

  loss = F.cross_entropy(logit, label_all_sentences)
  # print("loss",loss)
  # tensor(0.6406, device='cuda:0', grad_fn=<NllLossBackward>)

  loss.backward()
  model_API_obj.optimizer.step()

