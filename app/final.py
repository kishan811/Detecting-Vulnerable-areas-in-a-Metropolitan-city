#!/usr/bin/env python
# coding: utf-8

# In[36]:


import tweepy
from tweepy import OAuthHandler
import json
import datetime as dt
import time
import os
import sys
import sqlite3
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


# In[37]:





# In[38]:


keyWord = 'beautiful'


# In[166]:





# In[167]:


# In[42]:


import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs
from subprocess import check_output


# In[55]:


def loadModel():
    PATH = os.getcwd()
    filename =  'cnn_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


# In[56]:


loaded_model = loadModel()


# In[57]:


def stopwordsCreate():    
    nltk.download('stopwords')
    sns.set_style("whitegrid")
    np.random.seed(0)

    MAX_NB_WORDS = 100000
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    return stop_words, tokenizer, MAX_NB_WORDS


# In[58]:


def loadWordEmbedding():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('input/fasttext/wiki.simple.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index


# In[59]:


#model parameters
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4


# In[65]:


def tweetAnalysis(tweets, stop_words, tokenizer, embeddings_index, MAX_NB_WORDS):
    test_df = tweets
    test_df = test_df.fillna('_NA_')
    label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    raw_docs_test = test_df['cleaned_tweet'].tolist() 
#     raw_docs_test = [tweets,]
    num_classes = len(label_names)

    processed_docs_test = []
    for doc in tqdm(raw_docs_test):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs_test.append(" ".join(filtered))
    #end for

    print("tokenizing input data...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs_test)  #leaky
    word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    #pad sequences
    word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=168)
    
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return word_seq_test


# In[61]:


def predictionLabel(word_seq_test):
    y_test = loaded_model.predict(word_seq_test)
    labelList = []
    for val in y_test:
        if (val[0] * 100) <= 2:
            labelList.append('Normal')
        elif (val[0] * 100) > 2 and (val[0] * 100) <= 8:
            labelList.append('Less Harmful')
        else:
            labelList.append('Harmful')
    return labelList


# In[62]:


def CNNModel(tweets):
    stop_words, tokenizer, MAX_NB_WORDS = stopwordsCreate()
    embeddings_index = loadWordEmbedding()
    word_seq_test = tweetAnalysis(tweets, stop_words, tokenizer, embeddings_index, MAX_NB_WORDS)
    predictedLabel = predictionLabel(word_seq_test)
    return predictedLabel


# In[66]:


#predictedLabel = CNNModel(df)


# In[ ]:


def getGeoTagCount(df):
    count = 0
    for val in df[['latitude','longitude']]:
        if val[0] != 'NULL' and val[1] != 'NULL':
            count += 1
    return count

def countLabel(predictedLabel, tweetLabel):
    for val in predictedLabel:
        tweetLabel[val] += 1

def searchByKeyword(keyWord):
    conn = sqlite3.connect('twitter.db', isolation_level=None, check_same_thread=False)
    c = conn.cursor()
    df = pd.read_sql("SELECT * FROM hyd_prev_tweet_info where cleaned_tweet like '%" + keyWord + "%'    and (latitude != 'NULL' or longitude != 'NULL' or loc_from_tweet != 'NULL' or user_location != '')     ",conn)
    return len(df), df

def get_results(keyword):
    count, df = searchByKeyword(keyWord)
    predictedLabel = CNNModel(df)
    tweetLabel = {'Normal' : 0, 'Less Harmful' : 0, 'Harmful' : 0}
    countLabel(predictedLabel, tweetLabel)
    geoTagCount = getGeoTagCount(df)
    return count

print(get_results(sys.argv[1]))
