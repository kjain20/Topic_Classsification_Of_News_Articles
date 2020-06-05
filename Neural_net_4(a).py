
# coding: utf-8

import numpy as np
import pandas as pd
import random
from numpy import linalg as LA
import math
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import find
import tensorflow
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.datasets import fetch_rcv1
import matplotlib.pyplot as plt

rcv1 = fetch_rcv1()
A = rcv1['target']
CCAT = A[:,33]#.todense()
CCAT = CCAT.astype(np.int64)
Data = rcv1['data']
Training_data = Data[:100000]
Test_data = Data[100000:]
Training_label = CCAT[:100000]
Test_label = CCAT[100000:]
[articles, features] = Training_data.shape


np.random.seed(7)
model = Sequential()
model.add(Dense(100, input_dim=features, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
sgd=optimizers.SGD(lr=0.1,momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
history_1 = model.fit(Training_data, Training_label, epochs=5, batch_size=500,verbose=1)

np.random.seed(7)
model2 = Sequential()
model2.add(Dense(100, input_dim=features, activation='relu'))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
sgd=optimizers.SGD(lr=0.1,momentum=0.9)
model2.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
history_2 = model2.fit(Training_data, Training_label, epochs=5, batch_size=500,verbose=1)


np.random.seed(7)
model3 = Sequential()
model3.add(Dense(100, input_dim=features, activation='relu'))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
sgd=optimizers.SGD(lr=0.1,momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
history_3 = model.fit(Training_data, Training_label, epochs=5, batch_size=500,verbose=1)

error_1 = [1-a for a in history_1.history['acc']]
error_2 = [1-a for a in history_2.history['acc']]
error_3 = [1-a for a in history_3.history['acc']]
plt.plot(error_1, label = "Hidden layer: 1" )
plt.plot(error_2, label = "Hidden layer: 2" )
plt.plot(error_3, label = "Hidden layer: 3" )
plt.title('Model Error')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()


