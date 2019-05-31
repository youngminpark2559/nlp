# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/NLP/joosthub/chapters/chapter_5/5_2_CBOW && \
# rm e.l && python 5_2_munging_frankenstein.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import os

from argparse import Namespace
import collections
import nltk.data
import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm_notebook

# ================================================================================
args = Namespace(
    raw_dataset_txt="data/books/frankenstein.txt",
    window_size=5,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/books/frankenstein_with_splits_my_run.csv",
    seed=1337
)

# ================================================================================
# Split the raw text book into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# ================================================================================
with open(args.raw_dataset_txt) as fp:
    book = fp.read()

# ================================================================================
# print("book",book)
# Frankenstein,

# or the Modern Prometheus


# by

# Mary Wollstonecraft (Godwin) Shelley


# Letter 1


# St. Petersburgh, Dec. 11th, 17--

# TO Mrs. Saville, England

# You will rejoice to hear that no disaster has accompanied the
# commencement of an enterprise which you have regarded with such evil
# forebodings.  I arrived here yesterday, and my first task is to assure
# my dear sister of my welfare and increasing confidence in the success
# of my undertaking.

# I am already far north of London, and as I walk in the streets of
# Petersburgh, I feel a cold northern breeze play upon my cheeks, which
# braces my nerves and fills me with delight.  Do you understand this
# feeling?  This breeze, which has travelled from the regions towards
# which I am advancing, gives me a foretaste of those icy climes.
# Inspirited by this wind of promise, my daydreams become more fervent
# and vivid.  I try in vain to be persuaded that the pole is the seat of
# frost and desolation; it ever presents itself to my imagination as the
# region of beauty and delight.  There, Margaret, the sun is forever
# visible, its broad disk just skirting the horizon and diffusing a
# perpetual splendour.  There--for with your leave, my sister, I will put
# some trust in preceding navigators--there snow and frost are banished;
# and, sailing over a calm sea, we may be wafted to a land surpassing in
# wonders and in beauty every region hitherto discovered on the habitable
# globe.  Its productions and features may be without example, as the


sentences = tokenizer.tokenize(book)
# print("sentences",sentences)
# ['Frankenstein,\n\nor the Modern Prometheus\n\n\nby\n\nMary Wollstonecraft (Godwin) Shelley\n\n\nLetter 1\n\n\nSt. Petersburgh, Dec. 11th, 17--\n\nTO Mrs. Saville, England\n\nYou will rejoice to hear that no disaster has accompanied the\ncommencement of an enterprise which you have regarded with such evil\nforebodings.', 
#  'I arrived here yesterday, and my first task is to assure\nmy dear sister of my welfare and increasing confidence in the success\nof my undertaking.', 
#  'I am already far north of London, and as I walk in the streets of\nPetersburgh, I feel a cold northern breeze play upon my cheeks, which\nbraces my nerves and fills me with delight.', 
#  'Do you understand this\nfeeling?', 
#  'This breeze, which has travelled from the regions towards\nwhich I am advancing, gives me a foretaste of those icy climes.', 
#  'Inspirited by this wind of promise, my daydreams become more fervent\nand vivid.', 
#  'I try in vain to be persuaded 

# ================================================================================
# print ("len(sentences)",len(sentences))
# 3427

# ================================================================================
# Clean sentences
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

# ================================================================================
cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]
# print("cleaned_sentences",cleaned_sentences)
# ['frankenstein , or the modern prometheus by mary wollstonecraft godwin shelley letter st . petersburgh , dec . th , to mrs . saville , england you will rejoice to hear that no disaster has accompanied the commencement of an enterprise which you have regarded with such evil forebodings . ', 
#  'i arrived here yesterday , and my first task is to assure my dear sister of my welfare and increasing confidence in the success of my undertaking . ', 
#  'i am already far north of london , and as i walk in the streets of petersburgh , i feel a cold northern breeze play upon my cheeks , which braces my nerves and fills me with delight . ', 
#  'do you understand this feeling ? ', 
#  'this breeze , which has travelled from the regions towards which i am advancing , gives me 

# ================================================================================
# Global vars
MASK_TOKEN = "<MASK>"

# ================================================================================
flatten = lambda outer_list: [item for inner_list in outer_list for item in inner_list]

# ================================================================================
win_sz=args.window_size
# print("args.window_size",args.window_size)
# 5

before_flatten=[]
for sentence in tqdm_notebook(cleaned_sentences):
    # Split "sentence" by one whitespace
    # print("sentence.split(' ')",sentence.split(' '))
    # ['frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.', 'petersburgh', ',', 'dec', '.', 'th', ',', 'to', 'mrs', '.', 'saville', ',', 'england', 'you', 'will', 'rejoice', 'to', 'hear', 'that', 'no', 'disaster', 'has', 'accompanied', 'the', 'commencement', 'of', 'an', 'enterprise', 'which', 'you', 'have', 'regarded', 'with', 'such', 'evil', 'forebodings', '.', '']

    splited_senten=sentence.split(' ')
    
    # ================================================================================
    mask_tok=[MASK_TOKEN]*win_sz
    # print("mask_tok",mask_tok)
    # ['<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>']

    # ================================================================================
    input_from_one_senten=mask_tok+splited_senten+mask_tok
    # print("input_from_one_senten",input_from_one_senten)
    # ['<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.', 'petersburgh', ',', 'dec', '.', 'th', ',', 'to', 'mrs', '.', 'saville', ',', 'england', 'you', 'will', 'rejoice', 'to', 'hear', 'that', 'no', 'disaster', 'has', 'accompanied', 'the', 'commencement', 'of', 'an', 'enterprise', 'which', 'you', 'have', 'regarded', 'with', 'such', 'evil', 'forebodings', '.', '', '<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>']
    
    # ================================================================================
    num=win_sz*2+1
    # print("num",num)
    # 11

    out_from_nltk=list(nltk.ngrams(input_from_one_senten,num))
    # print("out_from_nltk",out_from_nltk)
    # [('<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus'), ('<MASK>', '<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by'), ('<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary'), ('<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft'), ('<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin'), ('frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley'), (',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter'), ('or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st'), ('the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.'), ('modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.', 'petersburgh'), ('prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.', 'petersburgh', ','), ('by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.', 'petersburgh', ',', 'dec'), ('mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.', 'petersburgh', ',', 'dec', '.'),
    
    # ================================================================================
    before_flatten.append(out_from_nltk)

# ================================================================================
windows=flatten(before_flatten)
# print("windows",windows)
# [('<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus'), ('<MASK>', '<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by'), ('<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary'), ('<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft'), ('<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin'), ('frankenstein', ',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley'), (',', 'or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter'), ('or', 'the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st'), ('the', 'modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.'), ('modern', 'prometheus', 'by', 'mary', 'wollstonecraft', 'godwin', 'shelley', 'letter', 'st', '.', 'petersburgh'), ('prometheus', 'by', 'mary', 

# windows = flatten([list(nltk.ngrams([MASK_TOKEN] * args.window_size + sentence.split(' ') +     [MASK_TOKEN] * args.window_size, args.window_size * 2 + 1))     for sentence in tqdm_notebook(cleaned_sentences)])

# ================================================================================
# @ Create CBOW data
data = []
for window in tqdm_notebook(windows):
    # print("window",window)
    # ('<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>', 'frankenstein', ',', 'or', 'the', 'modern', 'prometheus')

    # ================================================================================
    target_token = window[win_sz]
    # print("target_token",target_token)
    # frankenstein

    # ================================================================================
    context = []
    for i, token in enumerate(window):
        if (token == MASK_TOKEN) or (i == win_sz):
            continue
        else:
            context.append(token)

    # print("context",context)
    # [',', 'or', 'the', 'modern', 'prometheus']

    # ================================================================================
    joined_tok=' '.join(token for token in context)
    # print("joined_tok",joined_tok)
    # , or the modern prometheus

    data.append([joined_tok,target_token])
    
    # data.append([' '.join(token for token in context), target_token])

# ================================================================================
# print("data",data)
# [[', or the modern prometheus', 'frankenstein'], 
#  ['frankenstein or the modern prometheus by', ','], 
#  ['frankenstein , the modern prometheus by mary', 'or'], 
#  ['frankenstein , or modern prometheus by mary wollstonecraft', 'the'], 
#  ['frankenstein , or the prometheus by mary wollstonecraft godwin', 'modern'], 
#  ['frankenstein , or the modern by mary wollstonecraft godwin shelley', 'prometheus'], 
#  [', or the modern prometheus mary wollstonecraft godwin shelley letter', 'by'], 
#  ['or the modern prometheus by wollstonecraft godwin shelley letter st', 'mary'], 
#  ['the modern 

# ================================================================================
# Convert to dataframe
cbow_data = pd.DataFrame(data, columns=["context", "target"])

# ================================================================================
# Create split data
n = len(cbow_data)
def get_split(row_num):
    if row_num <= n*args.train_proportion:
        return 'train'
    elif (row_num > n*args.train_proportion) and (row_num <= n*args.train_proportion + n*args.val_proportion):
        return 'val'
    else:
        return 'test'

cbow_data['split']= cbow_data.apply(lambda row: get_split(row.name), axis=1)

# ================================================================================
cbow_data.head()

# ================================================================================
# Write split data to file
cbow_data.to_csv(args.output_munged_csv, index=False)
