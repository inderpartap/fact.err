# -*- coding: utf-8 -*-
# @Author: Najeeb Qazi
# @Date:   2019-11-20 20:03:21
# @Last Modified by:   najeebq
# @Last Modified time: 2019-11-23 18:25:25

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import string
import numpy as np
import os
import re
import sys

#assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import udf
spark = SparkSession.builder.appName('DF processing').getOrCreate()
#assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

stemmer = PorterStemmer()
nltk.download('wordnet')
nltk.download('stopwords')


def main():

	tokenizer = RegexpTokenizer(r'\w+')

	file_schema = types.StructType([
		types.StructField('id', types.IntegerType()),
		types.StructField('title', types.StringType(), metadata={"maxlength":2048}),
		types.StructField('text', types.StringType(),metadata={"maxlength":2048}),
		types.StructField('label', types.IntegerType())
		])


	fake_file_schema = types.StructType([
		types.StructField('uuid', types.StringType()),
		types.StructField('ord_in_thread', types.IntegerType()),
		types.StructField('author', types.StringType()),
		types.StructField('published', types.StringType()),
		types.StructField('title', types.StringType()),
		types.StructField('text', types.StringType()),
		types.StructField('language', types.StringType()),
		types.StructField('crawled', types.StringType()),
		types.StructField('site_url', types.StringType()),
		types.StructField('country', types.StringType()),
		types.StructField('domain_rank', types.StringType()),
		types.StructField('thread_title', types.StringType()),
		types.StructField('spam_score', types.DoubleType()),
		types.StructField('main_img_url', types.StringType()),
		types.StructField('replies_count', types.IntegerType()),
		types.StructField('participants_count', types.IntegerType()),
		types.StructField('likes', types.IntegerType()),
		types.StructField('comments', types.IntegerType()),
		types.StructField('shares', types.IntegerType()),
		types.StructField('type', types.StringType()),
		])


	test_schema = types.StructType([types.StructField('title', types.StringType()),types.StructField('text', types.StringType())])

	#fake_test_dataset = spark.read.csv('s3://projfakenews/fake-news-test',schema=test_schema,header=True)

	#'xa0e'
	
	fake_dataset = spark.read.format("csv").option("header", "true").option("delimiter", "\t").option("multiLine", "true").load("s3://projfakenews/fake")
	#fake_test_dataset.show()
	#fake_dataset.show()

	fake_dataset = fake_dataset.drop('ord_in_thread','thread_title','author','published','uuid','language','crawled','main_img_url','site_url','participants_count','replies_count','likes','comments','shares','type','spam_score','domain_rank','country')


	#scraped_fake_dataset = fake_test_dataset.select(functions.col("headline").alias("title"),functions.col("description").alias("text")) 

	#fake_appended_dataset =fake_dataset.union(fake_test_dataset)
	fake_appended_dataset =fake_dataset

	#1 unreliable, 0 reliable
	fake_appended_dataset = fake_appended_dataset.withColumn("label",functions.lit(1))


	#fake_appended_dataset.show()


	kaggle_train = spark.read.format("csv").option("header", "true").option("delimiter", "\t").option("multiLine", "true").load("s3://projfakenews/fake-news_kaggle/train.tsv")


	kaggle_train = kaggle_train.drop('author','id')

	appended_data = kaggle_train.union(fake_appended_dataset)


	tokenizer = RegexpTokenizer(r'\w+')


	#tokenizing the text
	tokenizer_udf = udf(tokenize_string, types.StringType())


	#cast as string and lower case
	appended_data = appended_data.withColumn("text",functions.col("text").cast("string"))
	appended_data = appended_data.select(functions.lower(functions.col("title")).alias("title"),functions.lower(functions.col("text")).alias("text"),'label')


	appended_data = appended_data.select(tokenizer_udf("title").alias("title"),tokenizer_udf("text").alias("text"),"label")


	appended_data.show()

	#downloading nltk stopwords  & wordnet
	#removing stop words
	remove_stopwords_udf = udf(remove_stopwords, types.StringType())

	appended_data = appended_data.select(remove_stopwords_udf("title").alias("title"),remove_stopwords_udf("text").alias("text"),"label")


	#lemmatize
	#word_lemmatizer_udf = udf(word_lemmatizer, types.StringType())

	#appended_data = appended_data.select(word_lemmatizer_udf("title").alias("title"),word_lemmatizer_udf("text").alias("text"),"label")


	appended_data.write.csv("s3://projfakenews/ProcessedDatawithoutStemming", mode='overwrite')
	#stemming
	word_stemmer_udf = udf(word_stemmer, types.StringType())

	appended_data = appended_data.select(word_stemmer_udf("title").alias("title"),word_stemmer_udf("text").alias("text"),"label")


	#appended_data = appended_data.rdd

	#appended_data = appended_data.map(lambda x: x)


	appended_data.show()

	appended_data.write.csv("s3://projfakenews/ProcessedData", mode='overwrite')

	#remove punctuations


def tokenize_string(x):
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(str(x))

def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct


def strip_extraCommas(line):
	print(line)
	line = line.rstrip(',')


def word_stemmer(text):
  stem_text = " ".join([stemmer.stem(i) for i in text])
  return stem_text

def remove_stopwords(text):
	stopwordsList = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
	emptystr=[]
	for w in text:
		if w not in stopwordsList:
			emptystr.append(w)
	#words = [w for w in text if w not in stopwords.words("english")]
	return emptystr
  
def word_lemmatizer(text):
	lemmatizer = WordNetLemmatizer()
	lem_text =[lemmatizer.lemmatize(i) for i in text]
	return lem_text


def word_stemmer(text):
	stem_text =" ".join([stemmer.stem(i) for i in text])
	return stem_text



if __name__ == '__main__':
    #fileinput = sys.argv[1]
    main()

