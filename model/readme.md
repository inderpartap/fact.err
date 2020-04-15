<<<<<<< HEAD
requirements- install tensorflow, pickle

to load model


- model = tf.keras.models.load_model('fakenewsClassifier')

--read test file 
testDF= pd.read_csv("test.csv")


--read tokenizer using the following command
with open('tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)

-- convert into sequence or word vector by converting into embeddings, using the given dimensions


--saved a dummy file for prediction 'TestPredict'


To predict - run :
-model.predict('TestPredict')
=======
<h2>Models</h2>

requirements- install tensorflow, pickle
to load model


#### To Test Multi-layer Perceptron Classifier Model

In Spark environment, with a data file matching the given schema
```sh
$ spark-submit MLPClassifier.py
```
#### For Keras model

in Spark environment, with a data file matching the given schema
```sh
$ python modelGenerationAndAnalysisKeras.py
```

>>>>>>> 8aad17778d182a78f497ccfa5e8a9f9d95093c66




