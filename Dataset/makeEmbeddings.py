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
#import tensorflow_hub as hub
import os
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
# from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Activation,Embedding,LSTM
from keras.utils import np_utils, generic_utils
from keras import optimizers

from elephas.utils.rdd_utils import to_simple_rdd
from elephas.ml_model import ElephasEstimator
from elephas.spark_model import SparkModel
#assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import udf
from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics


import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()


spark = SparkSession.builder.appName('DF processing').getOrCreate()
#assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

#from pyspark import SparkContext, SparkConf
#conf = SparkConf().setAppName('Elephas_App').setMaster('local[8]')
#sc = SparkContext(conf=conf)



#read Data
#make glove embeddings using Dict
#fit data
#make word vectors
#tf.enable_eager_execution()


def main():

	data_schema = types.StructType([types.StructField('title', types.StringType()),types.StructField('text', types.StringType()),types.StructField('label', types.IntegerType())])

	datadf = spark.read.csv("s3://projfakenews/ProcessedDatawithoutStemming",schema=data_schema)

	#embeddings_index = make_embeddings()

	#reading and tokenizing
	#vocabulary_size = 2000

	# tokenizer   = Tokenizer()
	#textDF = datadf.select("text")
	# tokenizer.fit_on_texts(textDF)
	# datadf.show()
	datadf.show()

	word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="text", outputCol="features")
	datadf = datadf.withColumn("text", functions.array("text"))
	model = word2Vec.fit(datadf)
	result = model.transform(datadf)

	result.show()

	temp = result.select("features").show(10)
	result.dropna()

	make_model(result)

	#save tokenizer 
	# with open('tokenizer.pickle', 'wb') as handle:
	# 	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	





def make_model(data):
	data.show()
	data = data.dropna()
	nb_classes = data.select("label").distinct().count()
	input_dim = len(data.select("features").first()[0])

	print(nb_classes,input_dim)

	model = Sequential()
	model.add(Embedding(input_dim=input_dim,output_dim=100))
	#model.add(LSTM(64,return_sequences=False,dropout=0.1,recurrent_dropout=0.1))
	model.add(Dense(100,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes,activation='softmax'))
	#sgd = optimizers.SGD(lr=0.1)
	#model.compile(sgd, 'categorical_crossentropy', ['acc'])
	model.compile(loss ='binary_crossentropy', optimizer='adam')

	#model.compile(loss='categorical_crossentropy', optimizer='adam')
	spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')

	
	

	adam = optimizers.Adam(lr=0.01)
	opt_conf = optimizers.serialize(adam)

	estimator = ElephasEstimator()
	estimator.setFeaturesCol("features")
	estimator.setLabelCol("label")
	estimator.set_keras_model_config(model.to_yaml())
	estimator.set_categorical_labels(True)
	estimator.set_nb_classes(nb_classes)
	estimator.set_num_workers(1)
	estimator.set_epochs(20) 
	estimator.set_batch_size(128)
	estimator.set_verbosity(1)
	estimator.set_validation_split(0.15)
	estimator.set_optimizer_config(opt_conf)
	estimator.set_mode("synchronous")
	estimator.set_loss("categorical_crossentropy")
	estimator.set_metrics(['acc'])


	#estimator = ElephasEstimator(model, epochs=20, batch_size=32, frequency='batch', mode='asynchronous', nb_classes=1)

	pipeline = Pipeline(stages=[estimator])
	#fitted_model = estimator.fit(data)
	#prediction = fitted_model.transform(data)
	
	fitted_pipeline = pipeline.fit(data) # Fit model to data
	prediction = fitted_pipeline.transform(data) # Evaluate on train data.
	# prediction = fitted_pipeline.transform(test_df) # <-- The same code evaluates test data.
	pnl = prediction.select("text", "prediction")
	pnl.show(100)

	prediction_and_label = pnl.map(lambda row: (row.text, row.prediction))
	metrics = MulticlassMetrics(prediction_and_label)
	print(metrics.precision())
	pnl = prediction.select("label", "prediction").show()
	pnl.show(100)

	#prediction_and_label= pnl.rdd.map(lambda row: (row.label, row.prediction))
	#metrics = MulticlassMetrics(prediction_and_label)
	#print(metrics.precision())
	#print(metrics.recall())
	#spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
	#spark_model.fit(data, np.array(labels), validation_split=0.4, epochs = 3)
	#model_glove.save('fakenewsClassifier') 

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
    #fileinput = sys.argv[1]
    main()



