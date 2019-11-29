import numpy as np
import pandas as pd 


import os
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

%matplotlib inline




def main():

	df = pd.read_csv('AppendedProcessedStopwords.csv')
	df= df.drop(columns= ['Unnamed: 0','Unnamed: 0.1'])
	df = df.dropna()

	x= df['text']
	labels = df['label']

	token = Tokenizer()
	token.fit_on_texts(x)
	seq = token.texts_to_sequences(x)

	pad_seq = pad_sequences(seq,maxlen=300)


	#if you want to save tokenizer
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)

	embedding_vector = {}
	f = open('glove.6B.300d.txt')
	for line in tqdm(f):
		value = line.split(' ')
		word = value[0]
		coef = np.array(value[1:],dtype = 'float32')
		embedding_vector[word] = coef


	make_model()




def make_model():
	#training model
	model = Sequential()
	model.add(Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = False))
	model.add(Bidirectional(CuDNNLSTM(75)))
	model.add(Dense(32,activation = 'relu'))
	model.add(Dense(1,activation = 'sigmoid'))
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])
	history = model.fit(pad_seq,labels,epochs = 5,batch_size=256,validation_split=0.2)	

	model.save("BILSTM_300")
	



if __name__ == '__main__':
    #fileinput = sys.argv[1]
    main()



