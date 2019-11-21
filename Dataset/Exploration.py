#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import string


# In[3]:


fake_dataset = pd.read_csv('fake-news/fake.csv')


# In[4]:


fake_dataset = fake_dataset.drop(columns=['thread_title','author','published','uuid','main_img_url','participants_count','likes','comments','shares','type','spam_score','domain_rank','country'])


# In[5]:


fake_dataset = fake_dataset.drop(columns=['language','crawled','replies_count','ord_in_thread'])


# In[6]:


fake_dataset = fake_dataset.drop(columns=['site_url'])


# In[7]:


fake_dataset


# In[8]:


scraped_fake_dataset = pd.read_csv('fake-news/test.csv')


# In[9]:


scraped_fake_dataset = scraped_fake_dataset.rename(columns={"headline":"title", "description":"text"}) 


# In[10]:


scraped_fake_dataset


# In[11]:


appended_fakedataset = fake_dataset.append(scraped_fake_dataset,ignore_index=True)


# In[12]:


appended_fakedataset


# In[13]:


#check for NAN and missing values
#save it in a csv file
#do pre processing using NLTK
#some kind of anaylysis using headline and shit
#convert into fastText vectors
#modeling






# In[14]:


appended_fakedataset.shape


# In[15]:


np.ones(13308)


# In[16]:


appended_fakedataset['label'] = np.ones(13308)


# In[17]:


appended_fakedataset


# In[18]:


kaggle_train = pd.read_csv('fake-news_kaggle/train.csv')


# In[19]:


kaggle_train = kaggle_train.drop(columns=['author','id'])


# In[20]:


kaggle_train['label'].sum()


# In[21]:


kaggle_train = kaggle_train.append(appended_fakedataset)


# In[22]:


kaggle_train


# In[23]:


kaggle_train.to_csv(r'Appended.csv')


# In[24]:


kaggle_train['label'].sum()


# In[25]:


def remove_punctuation(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct


# In[26]:


def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words("english")]
    return words


# In[50]:


nltk.download('stopwords')


# In[27]:


kaggle_train =kaggle_train.dropna()


# In[28]:


kaggle_train['label'].sum()


# In[31]:


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns


# In[32]:


kaggle_train['text'][0].tolist()[0]


# In[33]:


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)


# In[1]:compat.v1.tables_initializer





with tf.Session() as session:
  session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
  message_embeddings = session.run(embed(kaggle_train['text'][0].tolist()[0]))

  for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    #print("Message: {}".format(messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join(
        (str(x) for x in message_embedding[:3]))
    print("Embedding: [{}, ...]\n".format(message_embedding_snippet))


# In[ ]:





# In[38]:





# In[ ]:




