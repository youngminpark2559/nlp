# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

# /mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/V_00001/prj_root/src/utils_NLP/utils_find_similarity_of_docs_against_vocab_files.py

# ================================================================================
from PIL import Image
import PIL.ImageOps
import scipy.misc
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
from sklearn.model_selection import RepeatedKFold
import skimage
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle
import time,timeit,datetime
import sys,os
import glob
import natsort 
from itertools import zip_longest # for Python 3.x
import math
import traceback

from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

# ================================================================================
from src.networks import networks as networks

from src.utils import utils_image as utils_image
from src.utils import utils_common as utils_common
 
# ================================================================================
def calculate(dir_Docs,dir_Vocab,args):
  dir_Docs_files=utils_common.get_file_list(dir_Docs)
  dir_Vocab_files=utils_common.get_file_list(dir_Vocab)
  # print("dir_Docs_files",dir_Docs_files)
  # print("dir_Vocab_files",dir_Vocab_files)
  # ['/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Docs/Doc1.txt', '/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Docs/Doc2.txt', '/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Docs/Doc3.txt']
  # ['/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Vocab/Vocab_side_effect.txt', '/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Vocab/Vocab_symptom.txt']

  # ================================================================================
  stop_words=set(stopwords.words('english'))
  wnl=WordNetLemmatizer()

  # ================================================================================
  start=timeit.default_timer()

  all_docs=[]
  for one_doc in dir_Docs_files:
    doc_contents,num_lines=utils_common.return_path_list_from_txt(one_doc)
    doc_contents=list(map(lambda one_line:one_line.replace("\n",""),doc_contents))
    # print("doc_contents",doc_contents)
    # ['I have weight loss.', "I'm thirsty.", 'What is the problem?']

    doc_str=" ".join(doc_contents)
    # print("doc_str",doc_str)
    # I have weight loss. I'm thirsty. What is the problem?

    tokenizer=RegexpTokenizer(r'\w+')
    doc_str_wo_punctu=tokenizer.tokenize(doc_str)
    # print("doc_str_wo_punctu",doc_str_wo_punctu)
    # ['I', 'have', 'weight', 'loss', 'I', 'm', 'thirsty', 'What', 'is', 'the', 'problem']

    doc_str_wo_punctu=" ".join(doc_str_wo_punctu)

    word_tokens=word_tokenize(doc_str_wo_punctu)
    # print("word_tokens",word_tokens)
    # ['I', 'have', 'weight', 'loss', 'I', 'm', 'thirsty', 'What', 'is', 'the', 'problem']

    result=[]
    for w in word_tokens: 
      if w not in stop_words:
        result.append(w)

    # print("result",result)
    # ['I', 'weight', 'loss', 'I', 'thirsty', 'What', 'problem']

    sent=" ".join(result)
    # print("sent",sent)
    # I weight loss I thirsty What problem

    sent_after_lemma=" ".join([wnl.lemmatize(i) for i in sent.split()])
    # print("sent_after_lemma",sent_after_lemma)
    # I weight loss I thirsty What problem

    # ================================================================================
    doc_str=" ".join(result)
    # print("doc_str",doc_str)
    # I weight loss I thirsty What problem

    all_docs.append(doc_str)

  # ================================================================================
  vocab_files=["side_effect","symptom"]
  result=[]
  for vocab_file_idx,one_vocab_file in enumerate(dir_Vocab_files):
    one_vocab_contents,num_vocab=utils_common.return_path_list_from_txt(one_vocab_file)
    # print("one_vocab_contents",one_vocab_contents)
    # ['vomit\n', 'dizzy\n', 'diarrhea\n', 'fever\n', 'stomach gas']

    one_vocab_contents=list(map(lambda one_word:one_word.replace("\n",""),one_vocab_contents))
    # print("one_vocab_contents",one_vocab_contents)
    # ['vomit', 'dizzy', 'diarrhea', 'fever', 'stomach gas']

    one_vocab_contents=list(map(lambda one_word:wnl.lemmatize(one_word),one_vocab_contents))
    # print("one_vocab_contents",one_vocab_contents)

    # ================================================================================
    cnt_num_init=0
    for doc_idx,one_doc in enumerate(all_docs):
      # print("one_doc",one_doc)
      # I have weight loss. I'm thirsty. What is the problem?

      for one_word in one_vocab_contents:
        cnt_num=one_doc.count(one_word)
        cnt_num_init=cnt_num_init+cnt_num
        # print("cnt_num_init",cnt_num_init)
      result.append([vocab_files[vocab_file_idx],doc_idx+1,cnt_num_init])
      cnt_num_init=0

  # print("result",result)
  # Doc1: symptom, Doc2: side_effect, Doc3: symptom
  # [['side_effect', 1, 0], ['side_effect', 2, 3], ['side_effect', 3, 0], ['symptom', 1, 2], ['symptom', 2, 0], ['symptom', 3, 2]]

  stop=timeit.default_timer()
  took_time_sec=stop-start
  took_time_min=str(datetime.timedelta(seconds=took_time_sec))
  # print('took_time_min',took_time_min)
  # 0:00:01.516950

def color_word(dir_Docs,dir_Vocab,args):
  dir_Docs_files=utils_common.get_file_list(dir_Docs)
  dir_Vocab_files=utils_common.get_file_list(dir_Vocab)
  # print("dir_Docs_files",dir_Docs_files)
  # print("dir_Vocab_files",dir_Vocab_files)
  # ['/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Docs/Doc1.txt', '/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Docs/Doc2.txt', 
  #  '/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Docs/Doc3.txt']
  # ['/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Vocab/Vocab_side_effect.txt', 
  #  '/mnt/1T-5e7/Companies/Sakary/Management_by_files/00002_Architecture_specific_projects/00001_Classify_document_by_using_vocab_file/My_code/Data/Vocab/Vocab_symptom.txt']

  # ================================================================================
  stop_words=set(stopwords.words('english'))
  wnl=WordNetLemmatizer()

  # ================================================================================
  start=timeit.default_timer()

  all_docs=[]
  for one_doc in dir_Docs_files:
    doc_contents,num_lines=utils_common.return_path_list_from_txt(one_doc)
    doc_contents=list(map(lambda one_line:one_line.replace("\n",""),doc_contents))
    # print("doc_contents",doc_contents)
    # ['I have weight loss.', "I'm thirsty.", 'What is the problem?']

    doc_str=" ".join(doc_contents)
    # print("doc_str",doc_str)
    # I have weight loss. I'm thirsty. What is the problem?

    tokenizer=RegexpTokenizer(r'\w+')
    doc_str_wo_punctu=tokenizer.tokenize(doc_str)
    # print("doc_str_wo_punctu",doc_str_wo_punctu)
    # ['I', 'have', 'weight', 'loss', 'I', 'm', 'thirsty', 'What', 'is', 'the', 'problem']

    doc_str_wo_punctu=" ".join(doc_str_wo_punctu)

    word_tokens=word_tokenize(doc_str_wo_punctu)
    # print("word_tokens",word_tokens)
    # ['I', 'have', 'weight', 'loss', 'I', 'm', 'thirsty', 'What', 'is', 'the', 'problem']

    result=[]
    for w in word_tokens: 
      if w not in stop_words:
        result.append(w)

    # print("result",result)
    # ['I', 'weight', 'loss', 'I', 'thirsty', 'What', 'problem']

    sent=" ".join(result)
    # print("sent",sent)
    # I weight loss I thirsty What problem

    sent_after_lemma=" ".join([wnl.lemmatize(i) for i in sent.split()])
    # print("sent_after_lemma",sent_after_lemma)
    # I weight loss I thirsty What problem

    # ================================================================================
    doc_str=" ".join(result)
    # print("doc_str",doc_str)
    # I weight loss I thirsty What problem

    all_docs.append(doc_str)

  # ================================================================================
  all_docs_color=[]
  vocab_files=["side_effect","symptom"]
  vocab_files_to_color=["red","blue"]
  result=[]
  vocab_after_proc=[]
  for vocab_file_idx,one_vocab_file in enumerate(dir_Vocab_files):
    one_vocab_contents,num_vocab=utils_common.return_path_list_from_txt(one_vocab_file)
    # print("one_vocab_contents",one_vocab_contents)
    # ['vomit\n', 'dizzy\n', 'diarrhea\n', 'fever\n', 'stomach gas']

    one_vocab_contents=list(map(lambda one_word:one_word.replace("\n",""),one_vocab_contents))
    # print("one_vocab_contents",one_vocab_contents)
    # ['vomit', 'dizzy', 'diarrhea', 'fever', 'stomach gas']

    one_vocab_contents=list(map(lambda one_word:wnl.lemmatize(one_word),one_vocab_contents))
    # print("one_vocab_contents",one_vocab_contents)

    vocab_after_proc.append(one_vocab_contents)
  
  # print("vocab_after_proc",vocab_after_proc)
  # [['vomit', 'dizzy', 'diarrhea', 'fever', 'stomach gas'], ['weight loss', 'pee', 'thirsty', 'headache', 'hair loss']]

  # ================================================================================
  all_docs_colored=[]
  all_docs_blue_colored=[]
  cnt_num_init=0
  for doc_idx,one_doc in enumerate(all_docs):

    temp_red_str=[]
    first=True
    for one_word in vocab_after_proc[0]:
      tag_red='<font color="red">'+one_word+'</font>'

      if first==True:
        replaced_one=one_doc.replace(one_word,tag_red)
        # print("replaced_one",replaced_one)
        # I weight loss I thirsty What problem

        temp_red_str.append(replaced_one)
        first=False
      else:
        replaced_one=temp_red_str[0].replace(one_word,tag_red)
        # print("replaced_one",replaced_one)
        # I weight loss I thirsty What problem

        temp_red_str.pop(0)
        
        temp_red_str.append(replaced_one)
        # print("replaced_one",replaced_one)
    # print("temp_red_str",temp_red_str)
    all_docs_colored.append(temp_red_str[0])
    temp_red_str.pop(0)

    # ================================================================================
    temp_blue_str=[]
    first=True
    for one_word in vocab_after_proc[1]:
      tag_red='<font color="blue">'+one_word+'</font>'

      if first==True:
        replaced_one=one_doc.replace(one_word,tag_red)
        # print("replaced_one",replaced_one)
        # I weight loss I thirsty What problem

        temp_blue_str.append(replaced_one)
        first=False
      else:
        replaced_one=temp_blue_str[0].replace(one_word,tag_red)
        # print("replaced_one",replaced_one)
        # I weight loss I thirsty What problem

        temp_blue_str.pop(0)
        
        temp_blue_str.append(replaced_one)
        # print("replaced_one",replaced_one)
    # print("temp_blue_str",temp_blue_str)
    all_docs_blue_colored.append(temp_blue_str[0])
    temp_blue_str.pop(0)

  # ================================================================================
  all_docs_colored=list(map(lambda one_el:one_el+"<br>",all_docs_colored))
  all_docs_blue_colored=list(map(lambda one_el:one_el+"<br>",all_docs_blue_colored))

  # print("all_docs_colored",all_docs_colored)
  # print("all_docs_blue_colored",all_docs_blue_colored)
  # ['I weight loss I thirsty What problem<br>', 
  #  'I become <font color="red">dizzy</font> Meformin And I also <font color="red">diarrhea</font> <font color="red">stomach gas</font> Is side effect<br>', 
  #  'I head ache And I much pee weight loss symptom What<br>']

  # ['I <font color="blue">weight loss</font> I <font color="blue">thirsty</font> What problem<br>', 
  #  'I become dizzy Meformin And I also diarrhea stomach gas Is side effect<br>', 
  #  'I head ache And I much <font color="blue">pee</font> <font color="blue">weight loss</font> symptom What<br>']

  afaf
