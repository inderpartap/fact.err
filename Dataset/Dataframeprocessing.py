# -*- coding: utf-8 -*-
# @Author: Najeeb Qazi
# @Date:   2019-11-20 20:03:21
# @Last Modified by:   najeebq
# @Last Modified time: 2019-11-20 21:52:49


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns
import sys

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('DF processing').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
from pyspark.ml import PipelineModel



def main(fileinput):

	tokenizer = RegexpTokenizer(r'\w+')

	file_schema = types.StructType([
		types.StructField('id', types.IntegerType()),
		types.StructField('title', types.StringType(), metadata={"maxlength":2048}),
		types.StructField('text', types.StringType(),metadata={"maxlength":2048}),
		types.StructField('label', types.IntegerType())
    ])

	kaggle_train = spark.read.csv(fileinput, schema=file_schema,header=True)
	#kaggle_train = pd.read_csv(file)
	print(kaggle_train.count())
	kaggle_train =kaggle_train.dropna()

	kaggle_train.show()
	print(kaggle_train.count())


if __name__ == '__main__':
    fileinput = sys.argv[1]
    main(fileinput)

