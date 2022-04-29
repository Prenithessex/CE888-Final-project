#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

import tensorflow 

from tensorflow import keras



#train spider is training data
#train others is testing data


# In[47]:


#Load and read the dataset
df=pd.read_json('train_spider.json')
df


# In[48]:


#sample a small amount of data
df_sample= df.iloc[:-5999,:].copy()
df_sample


# In[49]:


###preprocessing


# In[50]:


#convert question coloumn into lower case
df_sample.question = df_sample.question.astype(str).str.lower()
df_sample


# In[51]:


#removing punctuations 
#df_sample["Question_NoPunctuation"] = df_sample['question'].str.replace('[^\w\s]','')
df_sample["Question_NoPunctuation"] = df_sample['question'].str.replace(r"[-()\"#/@;:<>{}`+=~|.!?,]", "")
df_sample["Question_NoPunctuation"] = df_sample['question'].str.replace(r"_"," ")
df_sample


# In[52]:


#tokenizing the unpunctuated coloumn
nltk.download('punkt')
index=0
df_sample["Question_NoPunc_tokened"]= df_sample['Question_NoPunctuation']
for i in df_sample['Question_NoPunctuation']:
    df_sample["Question_NoPunc_tokened"][index] = word_tokenize(i)
    index=index+1
df_sample


# In[53]:


#lemmetization
index=0
df_sample["lemmetized_text"]= df_sample['Question_NoPunc_tokened']
lemmatizer = WordNetLemmatizer()
k=0
for index in df_sample['Question_NoPunc_tokened']: 
    df_sample["lemmetized_text"][k] = ' '.join([lemmatizer.lemmatize(w) for w in index])
    k=k+1
df_sample


# In[54]:


#pos tagging
nltk.download('averaged_perceptron_tagger')
index=0
df_sample["pos_tagged"]= df_sample['Question_NoPunc_tokened']
lemmatizer = WordNetLemmatizer()
k=0
for index in df_sample['Question_NoPunc_tokened']: 
    df_sample["pos_tagged"][k] = nltk.pos_tag(index)
    k=k+1
df_sample


# In[55]:


#BOS tagging and EOS tagging

index=0
df_sample["tagged_text"]= df_sample['lemmetized_text']
bos = "<BOS> "
eos = " <EOS>"
k=0
for index in df_sample['Question_NoPunc_tokened']: 
    df_sample["tagged_text"][k] =  [bos + df_sample['lemmetized_text'][k] + eos]#for text in index
    k=k+1
df_sample


# In[75]:


#vocabulary tagging and Id'ing
from keras.preprocessing.text import Tokenizer
y=0
def vocab_creater(text_lists, VOCAB_SIZE):
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(text_lists[y])
    dictionary = tokenizer.word_index
  
    word2idx = {}
    idx2word = {}
    for k, v in dictionary.items():
        if v < VOCAB_SIZE:
            word2idx[k] = v
            index2word[v] = k
        if v >= VOCAB_SIZE-1:
            continue
          
    return word2idx, idx2word

word2idx[y], idx2word[y] = vocab_creater(text_lists[y]=df_sample["Question_NoPunctuation"][y]+df_sample["tagged_text"][y], VOCAB_SIZE=14999)
y=y+1


# In[87]:


#Tokenize Bag of words to Bag of IDs
from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 14999

def text2seq(encoder_text, decoder_text, VOCAB_SIZE):

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)
  
    return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences= text2seq(df_sample["Question_NoPunctuation"][y], df_sample["tagged_text"][y], VOCAB_SIZE)
y=y+1


# In[94]:


###padding(max_len)
from keras.preprocessing.sequence import pad_sequences

def padding(encoder_sequences, decoder_sequences, MAX_LEN):
  
    df_sample["Question_NoPunctuation"][y] = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    df_sample["tagged_text"][y] = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  
    return encoder_input_data, decoder_input_data

df_sample["Question_NoPunctuation"][y], df_sample["tagged_text"][y]= padding(encoder_sequences, decoder_sequences, MAX_LEN)
y=y+1


# In[ ]:


#Code Embedding
#Create Embedding Matrix from our Vocabulary

def embedding_matrix_creater(embedding_dimention):
  embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
  for word, i in word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector
  return embedding_matrix


# In[ ]:


#Create Embedding Layer
def embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix):
  
  embedding_layer = Embedding(input_dim = VOCAB_SIZE, 
                              output_dim = EMBEDDING_DIM,
                              input_length = MAX_LEN,
                              weights = [embedding_matrix],
                              trainable = False)
  return embedding_layer

embedding_layer = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix)


# In[ ]:


#Reshape the Data to a neural network shape
def decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):
  
  decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

  for i, seqs in enumerate(decoder_input_data):
      for j, seq in enumerate(seqs):
          if j > 0:
              decoder_output_data[i][j][seq] = 1.
  print(decoder_output_data.shape)
  
  return decoder_output_data

decoder_output_data = decoder_output_creater(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE)


# In[ ]:


#implementing Seq2Seq model
def seq2seq_model_builder(HIDDEN_DIM=300):
    
    df_sample["Question_NoPunctuation"] = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    df_sample["tagged_text"] = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    return model


# In[ ]:


#epoch
model = seq2seq_model_builder(HIDDEN_DIM=300)
model.summary()

