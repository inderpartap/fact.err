#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import re
import nltk
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    data = data.dropna(how="any")
    
    for col in ['title', 'text']:
        data[col] = data[col].apply(clean_sentence)
    
    return data


def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['title', 'text']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus


def tsne_plot(model):

    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],xy=(x[i], y[i]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
        #plotting only 100 words otherwise it will become diffficult to read, remove this condition to plot more words
        if i >100:
            break
    plt.show()



def main():


    #taking some random sample of data for visualizations
    data = pd.read_csv('News _dataset/Fake.csv').sample(5000, random_state=23)
    true_data = pd.read_csv('News _dataset/True.csv').sample(5000,random_state=23)

    #processing
    STOP_WORDS = nltk.corpus.stopwords.words()
    data = clean_dataframe(data)
    true_data = pd.read_csv('News _dataset/True.csv').sample(5000,random_state=23)
    true_data = clean_dataframe(true_data)


    #building corpus of words
    corpus = build_corpus(data)        


    #generating word2vec model so that we can plot the words, separately for fake news and true news
    model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
    true_corpus = build_corpus(true_data)        
    model.save("word2vecmodel")
    model_true = word2vec.Word2Vec(true_corpus, size=90, window=15, min_count=100, workers=4)
    model_true.save("word2vecmodel_true")


    #plotting 
    tsne_plot(model)
    tsne_plot(model_true)


if __name__ == "__main__":
    main()