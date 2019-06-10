# This is my personal study based on tutorial originated from
# https://nlp.gitbook.io/book/tf-with-kaggle/untitled/bag-of-words-meets-bags-of-popcorn/word2vec-jw

# ================================================================================
# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/Code_projects/NLP/Word2Vec/K_means/Today_code/Part00001 && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# ================================================================================
# header=0: 0 if first line of the file are column names
# quoting=3: ignore double quatation
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

# ================================================================================
# What you will do using CountVectorizer

# Sentence1: "The cat sat on the hat" 
# Sentence2: "The dog ate the cat and the hat"

# Collect unique words:
# {the, cat, sat, on, hat, dog, ate, and}

# Sentence1: {2, 1, 1, 1, 1, 0, 0, 0}
# Sentence1 has 2 "the", 1 "cat", ...
# Sentence1 is experessed by {2, 1, 1, 1, 1, 0, 0, 0} (this is also one of encoding methods)

# Sentence2: {3, 1, 0, 0, 1, 1, 1, 1}

# ================================================================================
# c vectorizer: CountVectorizer object
# max_features=5000: 5000 unique words in vocabulary set?
vectorizer=CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)

# ================================================================================
clean_train_reviews_np=np.array(clean_train_reviews)
# print("clean_train_reviews_np",clean_train_reviews_np.shape)
# (25000,)

# print("clean_train_reviews",clean_train_reviews)
# ['stuff going moment mj started listening music watching odd documentary watched wiz 
#   watched moonwalker maybe want get certain insight guy thought really cool 
#   eighties maybe make mind 

train_data_features=vectorizer.fit_transform(clean_train_reviews)
# print("train_data_features",train_data_features)
# (0, 2496)	1
# (0, 2122)	1
# (0, 4273)	1

# ================================================================================
# Each word is expressed by 5000 dimensional vector
train_data_features=train_data_features.toarray()
# print("train_data_features",train_data_features.shape)
# (25000, 5000)

# print("train_data_features",train_data_features)
# [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]

# ================================================================================
dist=np.sum(train_data_features,axis=0)
# print("dist",dist)
# [187 125 108 ... 740 518 147]

# 187: 0th word is used 187 times in entire text

# ================================================================================
# print("dist",len(dist))
# 5000

vocab_idx=vectorizer.vocabulary_
# print("vocab",vocab)
# {'stuff': 4267, 'going': 1905, 'moment': 2874, 'started': 4181, 'listening': 2590, 
#  'music': 2933,  'watching': 4834, 'odd': 3068, 'documentary': 1273, 
#  'watched': 4832, 'maybe': 2757, 'want': 

vocab_key=list(vocab_idx.keys())
vocab_index=list(vocab_idx.values())
# print("vocab_key",vocab_key)
# ['stuff', 'going', 'moment', 'started', 'listening', 'music', 'watching', 'odd', 'documentary', 

# print("vocab_index",vocab_index)
# [4267, 1905, 2874, 4181, 2590, 2933, 4834, 3068, 1273, 4832, 2757, 4809, 1873, 682, 2263, 1982, 

d={'vocab_index':vocab_index,'vocab_key':vocab_key}
idx_vocab_df=pd.DataFrame(d)
# print("idx_vocab_df",idx_vocab_df)
#       vocab_index    vocab_key
# 0            4267        stuff
# 1            1905        going

test=idx_vocab_df.sort_values(by=['vocab_index'],ascending=True)
# print("test",test)
# test       vocab_index     vocab_key
# 1723            0     abandoned
# 2521            1           abc
# ...
# 2685         4998       zombies
# 876          4999          zone

vocab_unsorted=vocab_idx.keys()
# print("vocab",vocab)
# dict_keys(['stuff', 'going', 'moment', 'started', 'listening', 'music', 'watching', 'odd', 

# print("vocab",len(vocab))
# 5000

vocab=test.iloc[:,1]
# print("vocab",vocab)
# 1723       abandoned
# 2521             abc
# 2326       abilities

# for tag,count in zip(vocab,dist):
#   print(count,tag)
# 187 abandoned
# 125 abc
# 108 abilities
# ...
# 740 zombie
# 518 zombies
# 147 zone

# ================================================================================
# Use "Random Forest classifier (100 trees)"
forest=RandomForestClassifier(n_estimators=100) 

# c train_X: 25000 sentences
train_X=train_data_features
# print("train_X",train_X)
# [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]

# c train_y: label for 25000 sentence
train_y=train["sentiment"]
# print("train_y",train_y)
# 0        1
# 1        1
# ...
# 24998    0
# 24999    1

# Find pattern betwen X and y
forest=forest.fit(train_X,train_y)

# ================================================================================
# @ Use "trained forest" on "test set"

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
# @ Convert each sentence into vector by using vectorizer

test_data_features=vectorizer.transform(clean_test_reviews)
test_data_features=test_data_features.toarray()
# print("test_data_features",test_data_features.shape)
# (25000, 5000)

# ================================================================================
# @ Use "trained tree classifier"

result=forest.predict(test_data_features)

# ================================================================================
data_for_df={"id":test["id"],"sentiment":result}
output=pd.DataFrame(data=data_for_df)

# ================================================================================
output.to_csv("./Result/Bag_of_Words_model.csv",index=False,quoting=3)
