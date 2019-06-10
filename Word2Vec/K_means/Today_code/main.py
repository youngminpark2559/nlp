# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/Code_projects/NLP/Word2Vec/K_means/Today_code && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# Clustering
# - It separate "data point" which element of each vector 
# into several groups by using some criterion like similarity

# ================================================================================
# Goal of clustering
# - You have N number of n dimeansional vectors
# - Elements in same vector has some similarity
# - Elements which are located in different vectors 
# has no explicit common relationship
# - Group data points which are located in N number of n dimensional vectors

# ================================================================================
# - n dimensional vector
# - If 3th dimension has high range of number compared to other dimension
# - you should perform "scaling"

# ================================================================================
# K-means clustering algorithm
# 1.- Randomly select K number of data points as K number of centroids
# 2.- Assign data points into the each nearest centroid
# 3.- Recalculate "new centroids"
# 4.- Iterate "2 and 3 steps" until centroids don't move

# ================================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from gensim.models import word2vec
from gensim.test.utils import common_texts,get_tmpfile

from nltk.corpus import stopwords
import nltk.data

# ================================================================================
from KaggleWord2VecUtility import KaggleWord2VecUtility

# ================================================================================
train=pd.read_csv('./Data/labeledTrainData.tsv',encoding='utf8',sep='\t')
# print("train",train)

# ================================================================================
sentences=[]
for review in train["review"]:
  sentences+=KaggleWord2VecUtility.review_to_sentences(review,remove_stopwords=False)

# print("sentences",np.array(sentences).shape)
# (266885,)

# ================================================================================
# @ Hyperparameters
num_features=300 # Number of dimension of vector which is composed of characters
min_word_count=40 # Number of words, minimal
num_workers=4 # Number of threads for parallel processing
context=10 # Contextual window size for string
downsampling=1e-3 # 문자 빈도 수 Downsample

# ================================================================================
# Train W2V model
model=word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,
  min_count=min_word_count,window=context,sample=downsampling)

# ================================================================================
# @ After compeleting the training, unload the useless memory
model.init_sims(replace=True)

# ================================================================================
# c model_name: file name of W2V trained checkpoint
model_name='./Models/300features_40minwords_10text'
# model_name='./Models/300features_50minwords_20text'

model.save(model_name)
afaf
# ================================================================================
# 모델도 만들어서 저장 했으니 이제 잘 구성되어 있는지 살펴보자.
# # 유사도가 없는 단어 추출
# model.wv.doesnt_match('man woman child kitchen'.split())




# model=Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
# model.save("~/word2vec.model")

# model=Word2Vec.load("~/word2vec.model")
# # print("model",model)
# # Word2Vec(vocab=12, size=100, alpha=0.025)



# # # 클러스터의 크기 "k"를 어휘 크기의 1/5이나 평균 5단어로 설정한다.
# # word_vectors = model.wv.syn0 # 어휘의 feature vector
# # num_clusters = word_vectors.shape[0] / 5
# # num_clusters = int(num_clusters)

# # # K means 를 정의하고 학습시킨다.
# # kmeans_clustering = KMeans( n_clusters = num_clusters )
# # idx = kmeans_clustering.fit_predict( word_vectors )

# # # 끝난 시간에서 시작시각을 빼서 걸린 시간을 구한다.
# # end = time.time()
# # elapsed = end - start
# # print("Time taken for K Means clustering: ", elapsed, "seconds.")
# # Time taken for K Means clustering: 252.84209394454956 seconds.

# # # 각 어휘 단어를 클러스터 번호에 매핑되게 word/Index 사전을 만든다.
# # idx = list(idx)
# # names = model.wv.index2word
# # word_centroid_map = {names[i]: idx[i] for i in range(len(names))}
# # #     word_centroid_map = dict(zip( model.wv.index2word, idx ))

# # # 첫 번째 클러스터의 처음 10개를 출력
# # for cluster in range(0,10):
# #     # 클러스터 번호를 출력
# #     print("\nCluster {}".format(cluster))

# #     # 클러스터 번호와 클러스터에 있는 단어를 찍는다.
# #     words = []
# #     for i in range(0,len(list(word_centroid_map.values()))):
# #         if( list(word_centroid_map.values())[i] == cluster ):
# #             words.append(list(word_centroid_map.keys())[i])
# #     print(words)
# # Cluster 0
# # ['terri', 'roy', 'noah', 'shawn', 'micheal', 'gilliam', 'mckenzi', 'ncis', 'costa', 'xavier', 'flaherti']

# # Cluster 1
# # ['indulg', 'conscious', 'absorb', 'loath', 'righteous', 'proclaim', 'esteem', 'wallow', 'reflex', 'referenti']

# # Cluster 2
# # ['coke', 'dope', 'drank']

# # Cluster 3
# # ['abu', 'cardin', 'kersey', 'pacifist', 'goliath', 'treason', 'tyrann']

# # Cluster 4
# # ['background', 'landscap', 'backdrop', 'vista', 'wildlif', 'travelogu']

# # Cluster 5
# # ['touch', 'poignant', 'profound', 'heartbreak', 'underst', 'uplift', 'heartfelt', 'heartwarm', 'sear']

# # Cluster 6
# # ['midnight', 'clock', 'pm', 'marathon']

# # Cluster 7
# # ['moor', 'patrick', 'lesli', 'barrymor', 'marc', 'lionel', 'carey', 'farrel', 'seymour', 'perkin', 'gale', 'stanton', 'dali', 'elisha', 'lacey', 'tyne']

# # Cluster 8
# # ['ann', 'mrs', 'juli', 'helen', 'susan', 'carol', 'elizabeth', 'drew', 'turner', 'alic', 'louis', 'kay', 'margaret', 'june', 'colbert', 'shelley', 'martha', 'beaver', 'kathleen', 'katherin', 'veronica', 'hayward', 'evelyn', 'judith', 'topper', 'fletcher', 'wither', 'claudett', 'delilah', 'jayn']

# # Cluster 9
# # ['data']

# # """
# # 판다스로 데이터프레임 형태의 데이터로 읽어온다.
# # QUOTE_MINIMAL (0), QUOTE_ALL (1), 
# # QUOTE_NONNUMERIC (2) or QUOTE_NONE (3).

# # 그리고 이전 튜토리얼에서 했던 것처럼 clean_train_reviews 와 
# # clean_test_reviews 로 텍스트를 정제한다.
# # """
# # train = pd.read_csv('data/labeledTrainData.tsv', 
# #                     header=0, delimiter="\t", quoting=3)
# # test = pd.read_csv('data/testData.tsv', 
# #                    header=0, delimiter="\t", quoting=3)
# # # unlabeled_train = pd.read_csv( 'data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )
# # from KaggleWord2VecUtility import KaggleWord2VecUtility
# # # 학습 리뷰를 정제한다.
# # clean_train_reviews = []
# # for review in train["review"]:
# #     clean_train_reviews.append(
# #         KaggleWord2VecUtility.review_to_wordlist( review, \
# #         remove_stopwords=True ))
# # # 테스트 리뷰를 정제한다.
# # clean_test_reviews = []
# # for review in test["review"]:
# #     clean_test_reviews.append(
# #         KaggleWord2VecUtility.review_to_wordlist( review, \
# #         remove_stopwords=True ))
# # # bags of centroids 생성
# # # 속도를 위해 centroid 학습 세트 bag을 미리 할당한다.
# # train_centroids = np.zeros((train["review"].size, num_clusters), \
# #     dtype="float32" )

# # train_centroids[:5]
# # array([[0., 0., 0., ..., 0., 0., 0.],
# # [0., 0., 0., ..., 0., 0., 0.],
# # [0., 0., 0., ..., 0., 0., 0.],
# # [0., 0., 0., ..., 0., 0., 0.],
# # [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

# # # centroid 는 두 클러스터의 중심점을 정의 한 다음 중심점의 거리를 측정한 것
# # def create_bag_of_centroids( wordlist, word_centroid_map ):

# #     # 클러스터의 수는 word / centroid map에서 가장 높은 클러스트 인덱스와 같다.
# #     num_centroids = max( word_centroid_map.values() ) + 1

# #     # 속도를 위해 bag of centroids vector를 미리 할당한다.
# #     bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

# #     # 루프를 돌며 단어가 word_centroid_map에 있다면
# #     # 해당하는 클러스터의 수를 하나씩 증가시켜 준다.
# #     for word in wordlist:
# #         if word in word_centroid_map:
# #             index = word_centroid_map[word]
# #             bag_of_centroids[index] += 1

# #     # bag of centroids를 반환한다.
# #     return bag_of_centroids
# # # 학습 리뷰를 bags of centroids로 변환한다.
# # counter = 0
# # for review in clean_train_reviews:
# #     train_centroids[counter] = create_bag_of_centroids( review, \
# #         word_centroid_map )
# #     counter += 1

# # # 테스트 리뷰도 같은 방법으로 반복해 준다.
# # test_centroids = np.zeros(( test["review"].size, num_clusters), \
# #     dtype="float32" )

# # counter = 0
# # for review in clean_test_reviews:
# #     test_centroids[counter] = create_bag_of_centroids( review, \
# #         word_centroid_map )
# #     counter += 1


# # # 랜덤포레스트를 사용하여 학습시키고 예측
# # forest = RandomForestClassifier(n_estimators = 100)

# # # train 데이터의 레이블을 통해 학습시키고 예측한다.
# # # 시간이 좀 소요되기 때문에 %time을 통해 걸린 시간을 찍도록 함
# # print("Fitting a random forest to labeled training data...")
# # %time forest = forest.fit(train_centroids, train["sentiment"])
# # Fitting a random forest to labeled training data...
# # CPU times: user 35.2 s, sys: 450 ms, total: 35.6 s
# # Wall time: 37.3 s

# # from sklearn.model_selection import cross_val_score
# # %time score = np.mean(cross_val_score(\
# #     forest, train_centroids, train['sentiment'], cv=10,\
# #     scoring='roc_auc'))
# # CPU times: user 4min 54s, sys: 3.77 s, total: 4min 58s
# # Wall time: 4min 30s

# # %time result = forest.predict(test_centroids)
# # CPU times: user 2.21 s, sys: 47.8 ms, total: 2.26 s
# # Wall time: 2.31 s

# # score
# # 0.91566112

# # # 결과를 csv로 저장
# # output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
# # output.to_csv("data/submit_BagOfCentroids_{0:.5f}.csv".format(score), index=False, quoting=3)
# # fig, axes = plt.subplots(ncols=2)
# # fig.set_size_inches(12,5)
# # sns.countplot(train['sentiment'], ax=axes[0])
# # sns.countplot(output['sentiment'], ax=axes[1])

# # png

# # output_sentiment = output['sentiment'].value_counts()
# # print(output_sentiment[0] - output_sentiment[1])
# # output_sentiment
# # 402

# # 0 12701
# # 1 12299
# # Name: sentiment, dtype: int64

# # # 캐글 점수 0.84908
# # print(330/528)
# # 0.625

# # 왜 이 튜토리얼에서는 Bag of Words가 더 좋은 결과를 가져올까?
# # 벡터를 평균화하고 centroids를 사용하면 단어 순서가 어긋나며 Bag of Words 개념과 매우 비슷하다. 성능이 (표준 오차의 범위 내에서) 비슷하기 때문에 튜토리얼 1, 2, 3이 동등한 결과를 가져온다.

# # 첫째, Word2Vec을 더 많은 텍스트로 학습시키면 성능이 좋아진다. Google의 결과는 10 억 단어가 넘는 코퍼스에서 배운 단어 벡터를 기반으로 한다. 학습 레이블이 있거나 레이블이 없는 학습 세트는 단지 대략 천팔백만 단어 정도다. 편의상 Word2Vec은 Google의 원래 C 도구에서 출력되는 사전 학습된 모델을 로드하는 기능을 제공하기 때문에 C로 모델을 학습 한 다음 Python으로 가져올 수도 있다.

# # 둘째, 출판된 자료들에서 분산 워드 벡터 기술은 Bag of Words 모델보다 우수한 것으로 나타났다. 이 논문에서는 IMDB 데이터 집합에 단락 벡터 (Paragraph Vector)라는 알고리즘을 사용하여 현재까지의 최첨단 결과 중 일부를 생성한다. 단락 벡터는 단어 순서 정보를 보존하는 반면 벡터 평균화 및 클러스터링은 단어 순서를 잃어버리기 때문에 여기에서 시도하는 방식보다 부분적으로 더 좋다.

# # 더 공부하기 : 스탠포드 NLP 강의 : Lecture 1 | Natural Language Processing with Deep Learning - YouTube