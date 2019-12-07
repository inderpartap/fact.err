
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

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import udf
from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics


import tensorflow.contrib.eager as tfe


spark = SparkSession.builder.appName('DF processing').getOrCreate()
#assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext




def main():

	data_schema = types.StructType([types.StructField('title', types.StringType()),types.StructField('text', types.StringType()),types.StructField('label', types.IntegerType())])

	datadf = spark.read.csv("s3://projfakenews/ProcessedData",schema=data_schema)

	datadf.show()

	word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="text", outputCol="features")
	datadf = datadf.withColumn("text", functions.array("text"))
	model = word2Vec.fit(datadf)
	result = model.transform(datadf)


	result.show()

	temp = result.select("features").show(1)

	result = result.dropna()
	result = result.randomSplit([0.8,0.2],24)
	print(result[0].count(),result[1].count())
	make_model(result[0],result[1])

	


def make_model(train,val):
	layers = [100, 100, 2]
	# create the trainer and set its parameters
	trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
	model = trainer.fit(train)
	result = model.transform(val)
	predictionAndLabels = result.select("prediction", "label")
	#predictionAndLabels.where(predictionAndLabels['prediction'] == 0 ).show()
	evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
	print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

	#save model
	mlp_path = "s3://projfakenews/mlp"
	model.save(mlp_path)



if __name__ == '__main__':
    #fileinput = sys.argv[1]
    main()



