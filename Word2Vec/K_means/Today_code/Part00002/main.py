# My study note which is originated from
# https://nlp.gitbook.io/book/tf-with-kaggle/untitled/bag-of-words-meets-bags-of-popcorn/word2vec-part2-4

# ================================================================================
# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/Code_projects/NLP/Word2Vec/K_means/Today_code/Part00002 && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import sys
root_path='/mnt/1T-5e7/Code_projects/NLP/Word2Vec/K_means/Today_code'
sys.path.insert(0,root_path)

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

# ================================================================================
from KaggleWord2VecUtility import KaggleWord2VecUtility

# ================================================================================
# Idea:
# - Suppose there are 3 neighboring words in 100 words-sentence
# - Those 3 words have similar meaning

# ================================================================================
train=pd.read_csv("../Data/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

review_data_in_train=train["review"]
# print("review_data_in_train",review_data_in_train.shape)
# (25000,)

# ================================================================================
def perform_preprocess_on_review_data_of_train(review_data_in_train):
  sentences=[]

  for review in review_data_in_train:
    processed=KaggleWord2VecUtility.review_to_sentences(review,remove_stopwords=False)
    # print("processed",processed)
    # [['with', 'all', 'this', 'stuff', 'go', 'down', 'at', 'the', 'moment', 'with', 'mj', 'i', 've', 'start', 
    #   'listen', 'to', 'his', 'music', 'watch', 'the', 'odd', 'documentari', 'here', 'and', 'there', 'watch', 
    #   'the', 'wiz', 'and', 'watch', 
    sentences+=processed
  
  return sentences

# sentences=perform_preprocess_on_review_data_of_train(review_data_in_train)

# ================================================================================
# Parameters of Word2Vec model

# - Architecture
#   - skip-gram (default): slow, better performance, good for large dataset
#   - CBOW: good for small dataset, predict "one" word from text

# - Learning algorithm
#   - Hierarchical softmax (default)
#   - Negative sampling

# - Downsampling for "too frequently showing words"
#   - Recommended value: [0.00001,0.001]

# - Dimenstion of vector for one word
#   - 300 in this tutorial

# - Size of context (window)
#   - Learning algorithm considers "context"
#   - 10 is used in this toturial

# - Worker threads: number of parallel processes, [4,6] are recommended

# - Minimal number of words
#   - If 40,
#   - all words which show less 40 times are not contained into word Embedding
#   - 40 is used for this tutorial

# ================================================================================
# @ Hyperparameters

# Dimension for one vector for one word
num_features=300
# Minimal number of words
min_word_count=40
# Worker threads
num_workers=4
# Size of context (window)
context=10
# Downsampling for "too frequently showing words"
downsampling=1e-3

# ================================================================================
def train_W2V_model_and_save_checkpoint_file(sentences):
  
  model=word2vec.Word2Vec(
    sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)

  # ================================================================================
  # After finishing the training, unload useless data from the memory
  # model.init_sims(replace=True)

  # ================================================================================
  # Checkpoint file name for trained W2V model
  model_name='./Models/300features_40minwords_10text'
  # model_name='./Models/300features_50minwords_20text'

  model.save(model_name)

# train_W2V_model_and_save_checkpoint_file(sentences)

# ================================================================================
model=Word2Vec.load('./Models/300features_40minwords_10text')
# model=gensim.models.Word2Vec('./Models/300features_40minwords_10text')
# model=gensim.models.Word2Vec.load('model')

# ================================================================================
sample_words='man woman child kitchen'.split()
# print("sample_words",sample_words)
# ['man', 'woman', 'child', 'kitchen']

# ================================================================================
abnormal_word=model.wv.doesnt_match(sample_words)
# print("abnormal_word",abnormal_word)
# kitchen

# ================================================================================
country_names="france england germany berlin".split()
# print("country_names",country_names)
# ['france', 'england', 'germany', 'berlin']

abnormal_word=model.wv.doesnt_match(country_names)
# print("abnormal_word",abnormal_word)
# berlin

# ================================================================================
most_similar_word=model.wv.most_similar("man")
# print("most_similar_word",most_similar_word)
# [('woman', 0.6616834402084351), 
#  ('doctor', 0.580306887626648), 
#  ('boy', 0.5510787963867188), 
#  ('millionair', 0.5457772016525269), 
#  ('priest', 0.5291855931282043), 
#  ('businessman', 0.5266597270965576), 
#  ('ladi', 0.5145279169082642), 
#  ('scientist', 0.5097665190696716), 
#  ('cop', 0.5004777908325195), 
#  ('son', 0.4818127751350403)]

# ================================================================================
# @ Visualization word Embedding array
# @ Perform dimensionality reduction using t-SNE
# https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim

model_name='./Models/300features_40minwords_10text'
model=g.Doc2Vec.load(model_name)

# ================================================================================
vocab=list(model.wv.vocab)
# print("vocab",vocab)
# ['with', 'all', 'this', 'stuff', 'go', 'down', 'at', 'the', 'moment', 'i', 've', 'start', 'listen', 'to',

# print("vocab",len(vocab))
# 6517

# ================================================================================
X=model[vocab]
# print("X",X)
# print("X",len(X))
# 6517

# print("X",type(X))
# <class 'numpy.ndarray'>
# print("X",np.array(X).shape)
# (6517, 300)
# 6517 words, each word is expressed by 300 dimension vector

# print("X[0][:10]",X[0][:10])
# [-2.0248723  -0.08599703  0.5914207  -0.5943042  -2.5347142   0.49954486 0.5423555   0.866892    1.521479   -0.05260603]

# ================================================================================
# @ High dimenstion vector ---> 2 dimension vector
tsne=TSNE(n_components=2)

# ================================================================================
number_of_words_to_be_visualized=300

X_tsne=tsne.fit_transform(X[:number_of_words_to_be_visualized,:])

# @ Visualization all words
# X_tsne=tsne.fit_transform(X)

df=pd.DataFrame(X_tsne,index=vocab[:number_of_words_to_be_visualized],columns=['x','y'])

# ================================================================================
fig=plt.figure()
fig.set_size_inches(1000,1000)
ax=fig.add_subplot(1,1,1)

ax.scatter(df['x'],df['y'])

for word,pos in df.iterrows():
  ax.annotate(word,pos,fontsize=10)
# plt.show()
# /home/young/Pictures/2019_06_11_10:03:20.png

# ================================================================================
def makeFeatureVec(words,model,num_features):
    """
    주어진 문장에서 단어 벡터의 평균을 구하는 함수
    """
    # Words in one review text
    # print("words",words)
    # ['with', 'all', 'this', 'stuff', 'go', 'down', 'at', 'the', 'moment', 'with', 'mj', 'i', 've', 'start', 
    #  'listen', 'to', 'his', 'music', 'watch', 'the', 'odd', 'documentari', 'here', 'and', 'there', 'watch', 
    #  'the', 'wiz', 'and', 'watch', 

    # print("words",np.array(words).shape)
    # (437,)

    # ================================================================================
    # (300,) zero 1D array
    featureVec=np.zeros((num_features,),dtype="float32")

    # ================================================================================
    nwords=0.

    # c all_ws_for_w2v: all words which are defined for W2V model
    all_ws_for_w2v=model.wv.index2word
    # print("all_ws_for_w2v",all_ws_for_w2v)
    # ['the', 'and', 'a', 'of', 'to', 'is', 'it', 'in', 'i', 'this', 'that', 's', 'movi', 'was', 'film', 
    #  'as', 'for', 'with', 'but', 't', 'you', 'on', 'be', 'not', 'have', 'he', 'are', 'his', 'one', 'all', 
    #  'at', 'they', 'like', 'by', 'an', 'who',

    # ================================================================================
    # Index2word는 모델의 사전에 있는 단어명을 담은 리스트이다.
    # 속도를 위해 set 형태로 초기화 한다.
    index2word_set=set(all_ws_for_w2v)

    converted_words=[]

    # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
    for word in words:
        # If one_word_of_review_text is contained in W2V_word_set
        if word in index2word_set:
            nwords = nwords + 1.

            # One word is converted into 300 dim vector
            converted_word=model[word]
            # print("converted_word",converted_word)
            # [-2.02487230e+00 -8.59970301e-02  5.91420710e-01 -5.94304204e-01
            #  -2.53471422e+00  4.99544859e-01  5.42355478e-01  8.66891980e-01

            # print("converted_word",converted_word.shape)
            # (300,)

            converted_words.append(converted_word)

            # featureVec,model[word]
            # featureVec = np.add()

    # ================================================================================
    converted_words_np=np.array(converted_words)
    # print("converted_words_np",converted_words_np)
    # [[-2.0248723  -0.08599703  0.5914207  ...  0.00737232  0.6321718 -0.21920455]
    #  [-0.39805543  0.66087127 -0.77703816 ...  1.2401668   1.4965681 0.67214227]

    # print("converted_words_np",converted_words_np.shape)
    # (413, 300)
    # 413 words in reivew text, one word is expressed by 300 dimendsional vector
    # 300 dimensional vector * 413

    # ================================================================================
    sum_all_word_vectors=np.sum(converted_words_np,axis=0)
    # print("sum_all_word_vectors",sum_all_word_vectors.shape)
    # (300,)

    # print("nwords",nwords)
    # 413.0
   
    average_vector_in_one_review_text=np.divide(sum_all_word_vectors,nwords)
    # print("average_vector_in_one_review_text",average_vector_in_one_review_text.shape)
    # (300,)
     
    return average_vector_in_one_review_text

def getAvgFeatureVecs(reviews, model, num_features):
    # print("reviews",reviews)
    # 0        [with, all, this, stuff, go, down, at, the, mo...
    # 1        [the, classic, war, of, the, world, by, timoth...
    # 2        [the, film, start, with, a, manag, nichola, be...

    num_reviews=len(reviews)
    # print("num_reviews",num_reviews)
    # 25000

    # print("num_features",num_features)
    # 300

    # ================================================================================
    counter=0.

    # c reviewFeatureVecs: 300 dimensional vector: average vector (average word) of one review text
    # c reviewFeatureVecs: average word vector (300 dimensional) * 25000 number of reviews
    # @ Create 2D array in advance to make processing speed fast
    reviewFeatureVecs=np.zeros((num_reviews,num_features),dtype="float32")
    # print("reviewFeatureVecs",reviewFeatureVecs.shape)
    # (25000, 300)
    
    for review in reviews:
       average_vector_of_one_review_text=makeFeatureVec(review,model,num_features)
       word_index=int(counter)
       reviewFeatureVecs[word_index,:]=average_vector_of_one_review_text

       counter=counter+1.

    return reviewFeatureVecs

def getCleanReviews(reviews):
  """
  Use multiple workers (multi threads)
  """
  clean_reviews=[]
  clean_reviews=KaggleWord2VecUtility.apply_by_multiprocessing(
    reviews["review"],KaggleWord2VecUtility.review_to_wordlist,workers=4)
  return clean_reviews

# ================================================================================
# print("train",train)
#               id                        ...                                                                     review
# 0       "5814_8"                        ...                          "With all this stuff going down at the moment ...
# 1       "2381_9"                        ...                          "\"The Classic War of the Worlds\" by Timothy ...

processed_train=getCleanReviews(train)
# print("processed_train",processed_train)
# 0        [with, all, this, stuff, go, down, at, the, mo...
# 1        [the, classic, war, of, the, world, by, timoth...
# 2        [the, film, start, with, a, manag, nichola, be...

# print("processed_train",np.array(processed_train).shape)
# processed_train (25000,)

trainDataVecs=getAvgFeatureVecs(processed_train,model,num_features)
# print("trainDataVecs",trainDataVecs)
# [[-0.09077143  0.22703806  0.09080739 ...  0.0924567   0.15991634
#   -0.00193545]
#  [-0.11477665  0.28611073  0.09533101 ...  0.31330475  0.21488433
#    0.07350203]

# print("trainDataVecs",np.array(trainDataVecs).shape)
# (25000, 300)

# ================================================================================
# n_estimators=100: 100 trees
# n_jobs=-1: use all threads
forest=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=2018)

# Dataset: [average_one_word_vector,sentiment_label]
forest=forest.fit(trainDataVecs,train["sentiment"])

# 10 folds cross validation
get_validation_scores=cross_val_score(forest,trainDataVecs,train['sentiment'],cv=10,scoring='roc_auc')
# print("get_validation_scores",get_validation_scores)
# [0.89554848 0.87485152 0.89953376 0.88378272 0.893272 0.89544352 0.894472 0.9064928 0.87455552 0.89039136]

avg_score=np.mean(get_validation_scores)
# print("avg_score",avg_score)
# 0.8908343679999999

# ================================================================================
# @ Prediction with test set

test=pd.read_csv("../Data/testData.tsv",header=0,delimiter="\t",quoting=3)

processed_test=getCleanReviews(test)

# c testDataVecs: one average (representive) word vector * number_of_review_text
testDataVecs=getAvgFeatureVecs(processed_test,model,num_features)

# Predict one sentiment from one average (representive) word vector
# But do it for all number of review text
result=forest.predict(testDataVecs)

output=pd.DataFrame(data={"id":test["id"],"sentiment":result})

output.to_csv('./Result/Word2Vec_AverageVectors_{0:.5f}.csv'.format(avg_score),index=False,quoting=3)
# When the pretrained model is 300features_40minwords_10text, accuracy score is 0.90709436799999987
# When the pretrained model is 300features_50minwords_20text, accuracy score is 0.86815798399999999

# Predicted sentiments
# Positive
# Negative
output_sentiment=output['sentiment'].value_counts()
# print("output_sentiment",output_sentiment)
# 0    12543
# 1    12457

# print(output_sentiment[0]-output_sentiment[1])
# 86
# There are 86 more reviews in 0 sentiment class

# ================================================================================
fig,axes=plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.countplot(train['sentiment'],ax=axes[0])
sns.countplot(output['sentiment'],ax=axes[1])
plt.show()
# /home/young/Pictures/2019_06_11_12:34:47.png