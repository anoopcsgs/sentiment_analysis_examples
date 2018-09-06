
# coding: utf-8

# In[1]:


from keras.datasets import imdb


(X_train, Y_train), (X_test, Y_test) = imdb.load_data()


# In[2]:


word_index = imdb.get_word_index()


# In[5]:


index_to_word = {val:key for (key, val) in word_index.items()}


# In[6]:


rev = ' '.join(index_to_word.get(val-3 , 'UNK') for val in X_train[1])
print(rev)

print(Y_train[1])


# In[8]:


import numpy as np
X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)
X.shape


# In[9]:


unique_words = np.unique(np.hstack(X_train))


# In[10]:


print(len(unique_words))
# print(X.shape)


# In[11]:


from keras.preprocessing import sequence
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)


# In[12]:


from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential

seed = 7
np.random.seed(seed)


# In[129]:


model = Sequential()
model.add(Embedding(88587, 32, input_length=500))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# In[170]:


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5)


# In[171]:


scores = model.evaluate(X_test, Y_test, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(scores)


# In[172]:


result = model.predict_classes(X_test)


# In[173]:


print(Y_test[21:58])
print(np.hstack(result[21:58]))


# In[125]:


# text = "big hair big boobs bad music and a giant safety pin these are the words to best describe this terrible movie i love cheesy horror movies and i've seen hundreds but this had got to be on of the worst ever made the plot is paper thin and ridiculous the acting is an abomination the script is completely laughable the best is the end showdown with the cop and how he worked out who the killer is it's just so damn terribly written the clothes are sickening and funny in equal measures the hair is big lots of boobs bounce men wear those cut tee shirts that show off their stomachs sickening that men actually wore them and the music is just synthesiser trash that plays over and over again in almost every scene there is trashy music boobs and paramedics taking away bodies and the gym still doesn't close for bereavement all joking aside this is a truly bad film whose only charm is to look back on the disaster that was the 80's and have a good old laugh at how bad everything was back then"
text = "bad management"

text_split = text.split()


# In[207]:


from nltk.corpus import stopwords
import re
text = "it was average value but didn't up to the mark"
text_split = text.lower().split()

text_to_index = [word_index[val]+3 if val in word_index else 0 for val in text_split ]
text_to_index = sequence.pad_sequences([text_to_index], 500)
res = model.predict_classes([text_to_index])
res


# In[169]:


from keras.layers import Dropout, Conv1D, MaxPooling1D, LSTM, Activation
model = Sequential()

# Embedding layer
model.add(Embedding(88587, 
                    50, 
                    input_length=500))
model.add(Dropout(0.25))

# Convolutional layer
model.add(Conv1D(64,
                5,
                padding='valid',
                activation='relu',
                strides=1))
model.add(MaxPooling1D(pool_size=4))

# LSTM layer
model.add(LSTM(70))

# Squash
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

