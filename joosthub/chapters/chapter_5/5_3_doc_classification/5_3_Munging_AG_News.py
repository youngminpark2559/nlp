# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/NLP/joosthub/chapters/chapter_5/5_3_doc_classification && \
# rm e.l && python 5_3_Munging_AG_News.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace

# ================================================================================
args = Namespace(
    raw_dataset_csv="data/ag_news/news.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/ag_news/news_with_splits_my_test.csv",
    seed=1337
)

# ================================================================================
# Read raw data
news = pd.read_csv(args.raw_dataset_csv, header=0)

# print("news",news.shape)
# (120000, 2)

# print("news.head()",news.head())
#    category                                              title
# 0  Business  Wall St. Bears Claw Back Into the Black (Reuters)
# 1  Business  Carlyle Looks Toward Commercial Aerospace (Reu...
# 2  Business    Oil and Economy Cloud Stocks' Outlook (Reuters)
# 3  Business  Iraq Halts Oil Exports from Main Southern Pipe...
# 4  Business  Oil prices soar to all-time record, posing new...

# ================================================================================
# Unique classes
uniq_category=set(news.category)
# print("uniq_category",uniq_category)
# {'Sci/Tech', 'Business', 'World', 'Sports'}

# ================================================================================
by_category = collections.defaultdict(list)

for _, row in news.iterrows():
    # print("row",row)
    # category                                             Business
    # title       Wall St. Bears Claw Back Into the Black (Reuters)

    # ================================================================================
    by_category[row.category].append(row.to_dict())

# ================================================================================
np.random.seed(args.seed)

cate_items=by_category.items()
# print("cate_items",cate_items)
# dict_items([('Business', [{'category': 'Business', 'title': 'Wall St. Bears Claw Back Into the Black (Reuters)'}, 
#                           {'category': 'Business', 'title': 'Carlyle Looks Toward Commercial Aerospace (Reuters)'},
#                           {'category': 'Business', 'title': "Oil and Economy Cloud Stocks' Outlook (Reuters)"}, 
#                           {'category': 'Business', 'title': 'Iraq Halts Oil Exports from Main Southern Pipeline 
final_list = []
for _, item_list in sorted(cate_items):
    np.random.shuffle(item_list)
    n = len(item_list)

    # ================================================================================
    n_train = int(args.train_proportion*n)
    n_val = int(args.val_proportion*n)
    n_test = int(args.test_proportion*n)
    
    # ================================================================================
    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'  
    
    # ================================================================================
    # Add to final list
    final_list.extend(item_list)

# ================================================================================
# Write split data to file
final_news = pd.DataFrame(final_list)
# print("final_news",final_news)
#         category                        ...                                                                      title
# 0       Business                        ...                                         Jobs, tax cuts key issues for Bush
# 1       Business                        ...                                       Jarden Buying Mr. Coffee #39;s Maker
# 2       Business                        ...                                          Retail sales show festive fervour


# ================================================================================
# print("final_news.split.value_counts()",final_news.split.value_counts())
# train    84000
# test     18000
# val      18000

# ================================================================================
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
    
final_news.title = final_news.title.apply(preprocess_text)
# print("final_news",final_news)
#         category                        ...                                                                      title
# 0       Business                        ...                                        jobs , tax cuts key issues for bush
# 1       Business                        ...                                          jarden buying mr . coffee s maker
# 2       Business                        ...                                          retail sales show festive fervour

# ================================================================================
# Write munged data to CSV
final_news.to_csv(args.output_munged_csv, index=False)
