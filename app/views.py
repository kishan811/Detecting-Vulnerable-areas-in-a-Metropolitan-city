from flask import render_template, session
from sqlalchemy import and_
from sqlalchemy import or_
from flask import url_for, redirect, request, make_response,flash
from tweepy import OAuthHandler
import datetime as dt
import time, os, sys, sqlite3, json, tweepy, re, pickle, pandas as pd, numpy as np, nltk, keras, csv, math, codecs
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
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
from subprocess import check_output
from keras import backend as K
import jsonpickle

from app import app

from geopy.geocoders import Nominatim

embeddings_index = {}
num_filters = 64 
embed_dim = 300 
weight_decay = 1e-4

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

########################## Functions ###################################
def loadModel():
	PATH = os.getcwd()
	filename =  'cnn_model.sav'
	pickle_file = open(filename, 'rb')
	loaded_model = pickle.load(pickle_file)
	pickle_file.close()
	return loaded_model

def stopwordsCreate(): 
	nltk.download('stopwords')
	sns.set_style("whitegrid")
	np.random.seed(0)

	MAX_NB_WORDS = 100000
	tokenizer = RegexpTokenizer(r'\w+')
	stop_words = set(stopwords.words('english'))
	stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
	return stop_words, tokenizer, MAX_NB_WORDS

def loadWordEmbedding(embeddings_index):
	f = codecs.open('input/fasttext/wiki.simple.vec', encoding='utf-8')
	for line in tqdm(f):
		values = line.rstrip().rsplit(' ')
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

def tweetAnalysis(tweets, stop_words, tokenizer, embeddings_index, MAX_NB_WORDS):
	test_df = tweets
	test_df = test_df.fillna('_NA_')
	label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
	raw_docs_test = test_df['cleaned_tweet'].tolist() 
	#     raw_docs_test = [tweets,]
	num_classes = len(label_names)
	tokenizer = RegexpTokenizer(r'\w+')
	processed_docs_test = []
	for doc in tqdm(raw_docs_test):
		tokens = tokenizer.tokenize(doc)
		filtered = [word for word in tokens if word not in stop_words]
		processed_docs_test.append(" ".join(filtered))
    #end for

	tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
	tokenizer.fit_on_texts(processed_docs_test)  #leaky
	word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
	word_index = tokenizer.word_index

	#pad sequences
	word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=168)

	#embedding matrix
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

def predictionLabel(word_seq_test):
	loaded_model = loadModel()
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

def CNNModel(tweets):
	stop_words, tokenizer, MAX_NB_WORDS = stopwordsCreate()
	word_seq_test = tweetAnalysis(tweets, stop_words, tokenizer, embeddings_index, MAX_NB_WORDS)
	predictedLabel = predictionLabel(word_seq_test)
	return predictedLabel

def getGeoTagCount(df):
	count = 0
	# for val in df[['latitude','longitude']]:
	# 	if val[0] != 'NULL' and val[1] != 'NULL':
	# 		count += 1
	i = 0
	for i in range(len(df)):
		if df['latitude'][i] != "NULL" and df['longitude'][i] != "NULL":
			print(i)
			count+=1
	return count

def countLabel(predictedLabel, tweetLabel):
	for val in predictedLabel:
		tweetLabel[val] += 1

def get_lat_long(location):
	geolocator = Nominatim(user_agent="College Project")
	location = geolocator.geocode(location)
	return location

def searchByKeyword(keyWord, preDate, postDate, City = 'hyd_tweet_info'):
	pat = os.getcwd()
	conn = sqlite3.connect(pat + '/twitter.db', isolation_level=None, check_same_thread=False)
	c = conn.cursor()
	if not keyWord == "" and (preDate == "" or postDate == ""):
		df = pd.read_sql("SELECT * FROM " + City + " where cleaned_tweet like '%" + keyWord + "%' and (latitude != 'NULL' or longitude != 'NULL' or loc_from_tweet != 'NULL' or user_location != '')     ",conn)
		return len(df), df
	if not keyWord == "" and not (preDate == "" or postDate == ""):
		df = pd.read_sql("SELECT * FROM " + City + " where cleaned_tweet like '%" + keyWord + "%' and (created_at > '"+ str(preDate) + "' and created_at <= '"+ str(postDate) +"') and (latitude != 'NULL' or longitude != 'NULL' or loc_from_tweet != 'NULL' or user_location != '')     ",conn)
		return len(df), df
	if keyWord == "" and not (preDate == "" or postDate == ""):
		query = "SELECT * FROM " + City + " where (created_at > '"+ str(preDate) +"' and created_at <= '"+ str(postDate) +"') and (latitude != 'NULL' or longitude != 'NULL' or loc_from_tweet != 'NULL' or user_location != '')"
		print(query)
		df = pd.read_sql(query ,conn)
		return len(df), df
	df = pd.read_sql("SELECT * FROM " + City + "",conn)
	return len(df), df

@app.route("/AnalysisByCity", methods=['GET', 'POST'])
def getFullData():
	K.clear_session()
	KeyWord, Date = "", ""
	keyWord = request.form['search']
	preDate = time.strftime('%a %b %d %H:%M:%S +0000 %Y', time.strptime('2019-03-11','%Y-%m-%d'))
	postDate = time.strftime('%a %b %d %H:%M:%S +0000 %Y', time.strptime('2019-03-12','%Y-%m-%d'))
	City = request.form['city']
	if City=="Hyderabad":
		City = 'hyd_tweet_info'
	elif City=="Banglore":
		City = 'blr_tweet_info'
	else:
		City = 'hyd_tweet_info'
	total_count, df = searchByKeyword(keyWord, preDate, postDate, City)
	predictedLabel = CNNModel(df)
	tweetLabel = {'Normal' : 0, 'Less Harmful' : 0, 'Harmful' : 0}
	countLabel(predictedLabel, tweetLabel)
	geoTagCount = getGeoTagCount(df)
	return getData(total_count, geoTagCount, df, predictedLabel, tweetLabel) 

def getData(total_count, geoTagCount, df, predictedLabel, tweetLabel):
	response_data = {}
	response_data["total_count"] = total_count
	response_data["geoTagCount"] = geoTagCount
	response_data["tweets"] = list(df['cleaned_tweet'])
	response_data["predictedLabel"] = predictedLabel
	response_data["Normal"] = tweetLabel["Normal"]
	response_data["Less Harmful"] = tweetLabel["Less Harmful"]
	response_data["Harmful"] = tweetLabel["Harmful"]
	# print("here 1")
	# print(df["loc_from_tweet"])
	# print("here 2")
	heatmap_list = []
	i = 0
	#print(type())
	print("*****************************************")
	# print(df["latitude"])
	for i in range(len(df)):
		if predictedLabel[i] == "Harmful":
			
			#print(type(df[i]["latitude"]))
			if df["latitude"][i] != "NULL":
				print(i)
				ls = [df["latitude"][i], df["longitude"][i], 1.0]
				heatmap_list.append(ls)
				# if df["loc_from_tweet"][i] != "NULL":
				# 	location = get_lat_long(df["loc_from_tweet"][i])
				# elif df["user_location"][i] != "NULL":
				# 	location = get_lat_long(df["user_location"][i])

				#ls = [location.latitude, location.longitude, 1.0]
			
	response_data["heatmap_list"] = heatmap_list		
	response_data = json.dumps(response_data)
	temp=response_data
	response_data = json.loads(response_data)
	return render_template("index.html", message=response_data,message2=temp)


if __name__ == '__main__':
	app.run(debug=True)
	app.run(host='1.1.0.0')
	loadWordEmbedding(embeddings_index)














































































































	