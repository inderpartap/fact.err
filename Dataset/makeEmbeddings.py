# -*- coding: utf-8 -*-
# @Author: Najeeb Qazi
# @Date:   2019-11-23 18:31:15
# @Last Modified by:   najeebq
# @Last Modified time: 2019-11-23 19:09:49
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import string
import numpy as np
import re
import sys
import tensorflow as tf
import tensorflow_hub as hub
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import udf
spark = SparkSession.builder.appName('DF processing').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext



#read Data
#make glove embeddings using Dict
#fit data
#make word vectors


def main(fileinput):

	data_schema = types.StructType([types.StructField('title', types.StringType()),types.StructField('text', types.StringType()),types.StructField('label', types.IntegerType())])

	datadf = spark.read.csv(fileinput,schema=data_schema)

	embeddings_index = make_embeddings()

	#reading and tokenizing
	vocabulary_size = 2000
	tokenizer = Tokenizer(num_words= vocabulary_size)

	textDF = datadf.select("text")

	tokenizer.fit_on_texts(textDF)
	datadf.show()

	tokenizer.word_index.items()

	embedding_matrix = np.zeros((vocabulary_size, 100))
	for word, index in tokenizer.word_index.items():
		if index > vocabulary_size - 1:
			break
		else:
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[index] = embedding_vector

	sequences = tokenizer.texts_to_sequences(kaggle_train['text'])

	#save tokenizer 
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	data = pad_sequences(sequences, maxlen=50)





def make_model():
	model_glove = Sequential()
	model_glove.add(Embedding(vocabulary_size, 50, input_length=50, weights=[embedding_matrix], trainable=False))
	model_glove.add(Dropout(0.2))
	model_glove.add(Conv1D(64, 5, activation='relu'))
	model_glove.add(MaxPooling1D(pool_size=4))
	model_glove.add(LSTM(100))
	model_glove.add(Dense(1, activation='sigmoid'))
	model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])## Fit train data
	model_glove.fit(data, np.array(labels), validation_split=0.4, epochs = 3)
	model_glove.save('fakenewsClassifier') 

def make_embeddings():
	embeddings_index = dict()
	f = open('glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index



if __name__ == '__main__':
    fileinput = sys.argv[1]
    main(fileinput)



