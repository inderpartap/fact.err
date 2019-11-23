from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import pickle




def main():
	data = makewordembeddings(title,body)
	pred = makePredictions(data)
	if pred[0][0] > 0.5:
		return (1,pred[0][0])
	else:
		return (0,pred[0][0])

	


def makewordembeddings(title,body):
	vocabulary_size = 2000
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	testDF = pd.DataFrame([title,body],columns=['title','text'])
	#title_sequences = tokenizer.texts_to_sequences(title)
	body_sequences = tokenizer.texts_to_sequences(testDF['text'])
	data = pad_sequences(body_sequences, maxlen=50)
	return data

def makePredictions(testdata):
	model_glove = tf.keras.models.load_model('fakenewsClassifier')
	pred = model_glove.predict(testdata)
	return pred


if __name__ == '__main__':
    main()


