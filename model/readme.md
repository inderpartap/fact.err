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




