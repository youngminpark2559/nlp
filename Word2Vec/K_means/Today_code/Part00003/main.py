# My study note which is originated from
# https://nlp.gitbook.io/book/tf-with-kaggle/untitled/bag-of-words-meets-bags-of-popcorn/word2vec-part2-4

# ================================================================================
# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/Code_projects/NLP/Word2Vec/K_means/Today_code/Part00003 && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import sys
root_path='/mnt/1T-5e7/Code_projects/NLP/Word2Vec/K_means/Today_code'
sys.path.insert(0,root_path)

import time,timeit,datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import gensim
from sklearn.manifold import TSNE
import matplotlib as mpl
# If "-" character is broken in the graph
mpl.rcParams['axes.unicode_minus']=False
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.models as g
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import nltk.data

# ================================================================================
from KaggleWord2VecUtility import KaggleWord2VecUtility

# ================================================================================
model=Word2Vec.load('./Models/300features_40minwords_10text')

# ================================================================================
# Saved Word2Vec can be used in various ways

wv_syn0=model.wv.syn0
# print("wv_syn0",wv_syn0)
# [[-0.4826763   1.3518528  -0.03162416 ...  0.21802689  0.45068264  0.5559675 ]
#  [ 0.14788812  0.40839374  0.6383199  ... -0.49818632  0.9279748  -0.29746133]

vocab=model.wv.vocab
# print("vocab",vocab)
# {'with': <gensim.models.keyedvectors.Vocab object at 0x7f52bb1d7b70>, 
#  'all': <gensim.models.keyedvectors.Vocab object at 0x7f52bb1d7c50>, 
#  'this': <gensim.models.keyedvectors.Vocab object at 0x7f52bb1d7cc0>, 
#  'stuff': <gensim.models.keyedvectors.Vocab object at 0x7f52bb1d7d30>,

index2word=model.wv.index2word
# print("index2word",index2word)
# ['the', 'and', 'a', 'of', 'to', 'is', 'it', 'in', 'i', 'this', 'that', 's', 'movi', 'was', 'film', 

wv=model.wv
# print("wv",wv)
# <gensim.models.keyedvectors.Word2VecKeyedVectors object at 0x7f52bd6bba20>

wv_with=model.wv['with'][:10]
# print("wv_with",wv_with)
# [-2.0248723  -0.08599703  0.5914207  -0.5943042  -2.5347142   0.49954486 0.5423555   0.866892    1.521479   -0.05260603]

# ================================================================================
start = time.time()

word_vectors=model.wv.syn0
# print("word_vectors.shape",word_vectors.shape)
# (6517, 300)

# One cluster will have 5 data points
# with creating 1303 clusters
num_clusters=word_vectors.shape[0]/5
# print("num_clusters",num_clusters)
# 1303.4

num_clusters=int(num_clusters)
# print("num_clusters",num_clusters)
# 1303

# ================================================================================
kmeans_clustering=KMeans(n_clusters=num_clusters)

# (6517, 300): (number_of_data_points,number_of_features)
idx=kmeans_clustering.fit_predict(word_vectors)

end=time.time()
elapsed=end-start
# print("elapsed:",elapsed,"seconds")
# elapsed: 67.23949337005615 seconds

# ================================================================================
# Create word:index dictionary, for mapping words to clusters

idx=list(idx)
# print("idx",idx)
# [810, 1117, 671, 632, 259, 753, 371, 551, 183, 392, 471, 411, 585, 514, 87, 554, 279, 385, 962, 38, 57, 

num_idx=len(idx)
# print("num_idx",num_idx)
# 6517
# - All words

num_uniq_idx=len(list(set(idx)))
# print("num_uniq_idx",num_uniq_idx)
# 1303
# - Number of clusters

# ================================================================================
names=model.wv.index2word
# print("names",names)
# ['the', 'and', 'a', 'of', 'to', 'is', 'it', 'in', 'i', 'this', 'that', 's', 'movi', 'was', 'film', 'as',

num_words=len(names)
# print("num_words",num_words)
# 6517

word_centroid_map={}
for i in range(num_words):
  one_word=names[i]
  # print("one_word",one_word)
  # the

  one_cluster_idx=idx[i]
  # print("one_cluster_idx",one_cluster_idx)
  # 849

  word_centroid_map[one_word]=one_cluster_idx

# word_centroid_map=dict(zip(model.wv.index2word,idx))

# ================================================================================
# Print some data points from the first 10 clusters

for cluster in range(0,10):
    # Cluster's number
    print("Cluster {}".format(cluster))
    # Cluster 0
    
    # ================================================================================
    words=[]

    word_centroid_map_values=list(word_centroid_map.values())
    # print("word_centroid_map_values",word_centroid_map_values)
    # [1033, 872, 780, 734, 397, 463, 465, 804, 276, 238, 527, 501, 753, 275, 87, 385, 462, 411, 399, 

    len_of_word_cent_map_vals=len(word_centroid_map_values)
    # print("len_of_word_cent_map_vals",len_of_word_cent_map_vals)
    # 6517

    word_centroid_map_keys=list(word_centroid_map.keys())
    # print("word_centroid_map_keys",word_centroid_map_keys)
    # ['the', 'and', 'a', 'of', 'to', 'is', 'it', 'in', 'i', 'this', 'that', 's', 'movi', 'was', 'film', 

    len_of_word_centroid_map_keys=len(word_centroid_map_keys)
    # print("len_of_word_centroid_map_keys",len_of_word_centroid_map_keys)
    # 6517

    # ================================================================================
    # Assign all 6517 number of words into corresponding clusters
    for i in range(0,len_of_word_cent_map_vals):
        cluster_idx_info_of_one_word=word_centroid_map_values[i]

        if cluster_idx_info_of_one_word==cluster:
            one_string_word=word_centroid_map_keys[i]
            # print("one_string_word",one_string_word)
            # japanes
            
            # Assign the word into each cluster list
            words.append(one_string_word)

    print("words",words)
    # Cluster 0
    # words ['interest']
    # Cluster 1
    # words ['quiet', 'nevertheless', 'semi', 'melodrama', 'bitter', 'raw', 'sophist', 'sinist', 'relax', 'immens', 'gentl', 'weight', 'astonish', 'tender', 'essenc', 'timeless', 'bold', 'awe', 'rivet', 'enchant', 'delici', 'poetic', 'strain', 'mixtur', 'heartbreak', 'stir', 'intim', 'honesti', 'nostalg', 'dazzl', 'startl', 'mesmer', 'uplift', 'engross', 'poetri', 'mundan', 'compliment', 'simplic', 'astound', 'provoc', 'undeni', 'warmth', 'vital', 'edgi', 'wrench', 'morbid', 'restrain', 'reson', 'delic', 'sustain', 'flair', 'keen', 'pale', 'rhythm', 'ingeni', 'transcend', 'sublim', 'enthral', 'heartfelt', 'steadi', 'brood', 'spontan', 'claustrophob', 'stimul', 'effortless', 'sensual', 'fragil', 'vibrant', 'heartwarm', 'energet', 'goldsworthi', 'subt', 'macabr', 'intrus', 'detach', 'optimist', 'peculiar', 'styliz', 'classi', 'offbeat', 'fuzzi', 'screwbal', 'textur', 'heighten', 'patho', 'artistri', 'unconvent', 'bittersweet', 'dreami', 'deft', 'taut', 'melancholi', 'quaint', 'upbeat', 'ambianc', 'seamless', 'lightheart', 'interlud']
    # Cluster 2
    # words ['mere', 'minor', 'consist', 'essenti', 'proper', 'invent', 'necessari', 'signific', 'convent', 'virtual', 'sourc', 'construct', 'consider', 'inevit', 'vagu', 'distinct', 'supernatur', 'subplot', 'method', 'exagger', 'multipl', 'relev', 'render', 'dynam', 'alter', 'vast', 'ironi', 'reduc', 'illustr', 'coher', 'shift', 'albeit', 'innov', 'precis', 'resolut', 'meaning', 'elabor', 'philosoph', 'transit', 'superfici', 'metaphor', 'ambigu', 'multi', 'subtleti', 'divers', 'vari', 'evok', 'slightest', 'plausibl', 'thread', 'dimens', 'devoid', 'compens', 'arc', 'unrel', 'crucial', 'incorpor', 'exposit', 'straightforward', 'weav', 'secondari', 'intric', 'linear', 'cohes', 'themat', 'intertwin']
    # Cluster 3
    # words ['georg', 'bill', 'j', 'jerri', 'dick']
    # Cluster 4
    # words ['ghost', 'witch', 'cinderella', 'fairi', 'princess']
    # Cluster 5
    # words ['loos', 'toy', 'bag', 'bone', 'shoe', 'rage', 'fist', 'closet']
    # Cluster 6
    # words ['stone', 'princ', 'notabl', 'cook', 'hall', 'winter', 'loui', 'rival', 'twin', 'guest', 'laura', 'ward']
    # Cluster 7
    # words ['isn']
    # Cluster 8
    # words ['send', 'sell']
    # Cluster 9
    # words ['fell']

# ================================================================================
train=pd.read_csv("../Data/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

# ================================================================================
# print("train",train.shape)
# (25000, 3)

# print("train",train.columns)
# [' id', 'sentiment', 'review']

# ================================================================================
review_0=train["review"][0]
example1=BeautifulSoup(review_0,"html5lib")
# print("example1",example1)
# <html><head></head><body>"With all this stuff going down at the moment with MJ i've started listening to his

example1_txt=example1.get_text()
# print("example1_txt",example1_txt)
# "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd

# ================================================================================
#[^a-zA-Z]: Find characters which are not "a-zA-Z", for example 5, <
# sub("[^a-zA-Z]", " "): non-a-zA-Z characters are converted into empty space
letters_only=re.sub("[^a-zA-Z]"," ",example1.get_text())
# print(letters_only)
# With all this stuff going down at the moment with MJ i ve started listening to

# ================================================================================
lower_case=letters_only.lower()
# print("lower_case",lower_case)
# with all this stuff going down at the moment with mj i ve started listening to

# ================================================================================
words=lower_case.split(" ")
# print("words",words)

# ================================================================================
nltk.download('stopwords')

stopwords_from_NLTK=stopwords.words('english')
# print("stopwords_from_NLTK",stopwords_from_NLTK)
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
#  "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 

# ================================================================================
# Non-stop words 

nonstop_words=[w for w in words if not w in stopwords.words("english")]
# print("nonstop_words",nonstop_words)
# ['', 'stuff', 'going', 'moment', 'mj', 'started', 'listening', 'music', '', 'watching', 'odd', 

# ================================================================================
def review_to_words(raw_review):
  # 1. Remove HTML tags from the text
  review_text=BeautifulSoup(raw_review,"html5lib").get_text()

  # 2. Replace all characters (which are not alphabet) with white space
  letters_only=re.sub("[^a-zA-Z]"," ",review_text)
  
  # 3. Lowercase, and split
  words=letters_only.lower().split()
  
  # 4. Searching in set is faster than searching in list
  stops = set(stopwords.words("english"))

  # 5. Remove stopwords
  meaningful_words=[w for w in words if not w in stops]
  
  # 6. Split
  out=" ".join(meaningful_words)
  
  return out

# ================================================================================
num_reviews=train["review"].size
# print("num_reviews",num_reviews)
# 25000

# ================================================================================
clean_train_reviews=[]
for i in range(0,num_reviews):
  one_review_text=train["review"][i]
  # print("one_review_text",one_review_text)
  # "With all this stuff going down at the moment with MJ i've started listening to his music, 
  #  watching the odd documentary here and there, 
  #  watched The Wiz and watched Moonwalker again. Maybe 
  
  one_rt_processed=review_to_words(one_review_text)
  # print("one_rt_processed",one_rt_processed)
  # stuff going moment mj started listening music watching odd documentary watched 
  # wiz watched moonwalker maybe want get certain insight guy thought 
  # really cool eighties maybe make mind 

  clean_train_reviews.append(one_rt_processed)

# print("clean_train_reviews",clean_train_reviews)
# print("clean_train_reviews",np.array(clean_train_reviews).shape)
# (25000,)

# ================================================================================
def create_bag_of_centroids(wordlist,word_centroid_map):

    word_cent_map_vals=word_centroid_map.values()
    # print("word_cent_map_vals",word_cent_map_vals)
    # dict_values([967, 964, 486, 542, 477, 513, 347, 635, 72, 288, 273, 705, 99, 271, 994, 522, 362, 
    #              412, 212, 38, 214, 455, 158, 393, 189, 63, 44, 317, 498, 557, 66, 243, 215, 202, 
    #              166, 184, 423, 146, 382, 114, 310, 541, 272,

    # print(len(list(word_cent_map_vals)))
    # 6517

    highest_cluster_idx=max(word_cent_map_vals)
    # print("highest_cluster_idx",highest_cluster_idx)
    # 1302

    num_centroids=highest_cluster_idx+1
    # print("num_centroids",num_centroids)
    # 1303

    # ================================================================================
    bag_of_centroids=np.zeros(num_centroids,dtype="float32" )

    # ================================================================================
    # print("wordlist",wordlist)
    # stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker 
    # maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent 
    # moonwalker part biography part feature film remember going see cinema originally released subtle messages 
    # mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson 
    # unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans 
    # would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal 
    # sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah 
    # joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things 
    # like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy 
    # bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom 
    # line movie people like mj one level another think people stay away try give wholesome message ironically mj 
    # bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention 
    # gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy 
    # one sickest liars hope latter
    
    for word in wordlist:
        if word in word_centroid_map:
            index=word_centroid_map[word]
            bag_of_centroids[index]+=1
    
    # ================================================================================
    # print("bag_of_centroids",bag_of_centroids)
    # [ 0. 35.  0. ...  0.  0.  0.]

    # stuff: cluster0?
    # going: cluster35?

    return bag_of_centroids

# ================================================================================
train_centroids=np.zeros((train["review"].size,num_clusters),dtype="float32" )

counter=0
for review in clean_train_reviews:
    # print("review",review)
    # ['monkey', 'dinosaur', 'hippi', 'bush', 'fighter', 'rap', 'puppet', 'pig', 'babe', 'wolf', 'jet', 
    #  'superhero', 'skit', 'punk', 'ninja', 'clown', 'wrestl', 'hop', 'ranger', 'holi', 'fever', 'blade', 
    #  'mummi', 'li', 'crocodil', 'airplan', 'breed', 'spider', 'riot', 'fri', 'rider', 'rocki', 'martian', 
    #  'chicken', 'bay', 'cannon', 'phantom', 'duck', 'cyborg', 'pink', 'predat', 'stalker', 'bunni', 'mighti', 
    #  'wax', 'geek', 'sniper', 'chainsaw', 'prank', 'hulk', 'circus', 'bread', 'duel', 'mayhem', 'biker', 'cow', 
    #  'dwarf', 'joker', 'deer', 'clad', 'redneck', 'frog', 'chip', 'creek', 'appl', 'cape', 'roar', 'fruit', 
    #  'monti', 'turtl', 'mice', 'soylent', 'hybrid', 'grass', 'hi', 'werewolv', 'chamber', 'blaze', 'insect', 
    #  'midget', 'lethal', 'glove', 'karat', 'thunder', 'claw', 'python', 'eater', 'stuf', 'jock', 'goat', 
    #  'gorilla', 'lamb', 'snowman', 'trooper', 'gigant', 'lizard', 'frozen', 'commando', 'hillbilli', 'bloke', 
    #  'krueger', 'skirt', 'armor', 'croc', 'fur', 'cradl', 'laser', 'brawl', 'gladiat', 'weirdo', 'dungeon', 
    #  'chocol', 'pickup', 'allig', 'sabretooth']
    # - All words in one review text

    # print("word_centroid_map",word_centroid_map)
    # {'the': 879, 'and': 787, 'a': 577, 'of': 542, 'to': 123, 'is': 291, 'it': 465, 'in': 475, 'i': 189, 
    #  'this': 344, 'that': 202, 's': 718, 'movi': 875, 'was': 346, 'film': 70, 'as': 597, 'for': 304, 
    #  'with': 381, 'but': 586, 't': 25, 'you': 224, 'on': 338, 'be': 117, 'not': 182, 'have': 130, 'he': 176, 
    #  'are': 69, 'his': 178, 'one': 312, 'all': 554, 'at': 280, 'they': 24, 'like': 290, 'by': 392, 'an': 55, 
    #  'who': 302, 'so': 323, 'from': 617, 'there': 209, 'her': 225, 'or': 192, 'just': 245, 'about': 314, 
    #  'out': 164, 'if': 105, 'has': 61, 'what': 434, 'time': 131, 'some': 399, 
    # W2V_words:cluster_idx ?

    out=create_bag_of_centroids(review,word_centroid_map)
    train_centroids[counter]=out

    counter+=1

# print("train_centroids",train_centroids)
# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]

# print("train_centroids",train_centroids.shape)
# (25000, 1303)

# print("train_centroids",np.unique(train_centroids))
# [0.000e+00 1.000e+00 2.000e+00 ... 2.073e+03 2.174e+03 3.075e+03]

# ================================================================================
test=pd.read_csv("../Data/testData.tsv",header=0,delimiter="\t",quoting=3)

# print("test",test.shape)
# (25000, 2)

# print("test.columns",test.columns)
# id,review

# ================================================================================
review_text_in_test=test["review"]
num_reviews=len(review_text_in_test)
# print("num_reviews",num_reviews)
# 25000

# ================================================================================
# @ Preprocess "review_text_in_test"

clean_test_reviews=[] 

for i in range(0,num_reviews):
  one_review_txt_in_test=review_text_in_test[i]
  clean_review=review_to_words(one_review_txt_in_test)
  clean_test_reviews.append(clean_review)

# ================================================================================
test_centroids=np.zeros((test["review"].size,num_clusters),dtype="float32")

# ================================================================================
counter=0
for review in clean_test_reviews:
    test_centroids[counter]=create_bag_of_centroids(review,word_centroid_map)
    counter+=1

# print("test_centroids",test_centroids)
# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]

# print("test_centroids",test_centroids.shape)
# (25000, 1303)

# print("test_centroids",np.unique(test_centroids))
# [0.000e+00 1.000e+00 2.000e+00 ... 2.562e+03 2.576e+03 2.816e+03]

# ================================================================================
# Use random foreset

forest=RandomForestClassifier(n_estimators=100)

forest=forest.fit(train_centroids,train["sentiment"])

result=forest.predict(test_centroids)
