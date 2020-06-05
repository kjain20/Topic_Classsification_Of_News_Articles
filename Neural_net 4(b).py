import numpy as np
from scipy.sparse import csr_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.datasets import fetch_rcv1
from tensorflow.keras import regularizers
from datetime import datetime

real_start = datetime.now()


rcv1 = fetch_rcv1()

data = csr_matrix(rcv1["data"])
target = rcv1["target"].toarray()

# CREATING LABELS
labels = np.add(np.zeros((target.shape[0])),0)
labels[np.where(target[:,33]==1)[0]] += 1

X_train = data[:100000]
y_train = labels[:100000]
X_test = data[100000:]
y_test = labels[100000:]


hidden_nodes = [50,100,150,200,300]
k = 500
n_iter = 5

final_acc = []
time = []

#This one is for 4 Hidden Layers
for h in hidden_nodes:
    start = datetime.now()
    model = Sequential()
    model.add(Dense(h, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=n_iter, batch_size=k,verbose=1) 
    end = datetime.now()
    time.append(end-start)
    final_acc.append(history.history["acc"][-1])
    
#This one is for 3 Hidden Layers
for h in hidden_nodes:
    start = datetime.now()
    model = Sequential()
    model.add(Dense(h, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=n_iter, batch_size=k,verbose=1) 
    end = datetime.now()
    time.append(end-start)
    final_acc.append(history.history["acc"][-1])

#This one is for 2 Hidden Layers    
for h in hidden_nodes:
    start = datetime.now()
    model = Sequential()
    model.add(Dense(h, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=n_iter, batch_size=k,verbose=1) 
    end = datetime.now()
    time.append(end-start)
    final_acc.append(history.history["acc"][-1])

#This one is for 1 Hidden Layer    
for h in hidden_nodes:
    start = datetime.now()
    model = Sequential()
    model.add(Dense(h, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(h, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=n_iter, batch_size=k,verbose=1) 
    end = datetime.now()
    time.append(end-start)
    final_acc.append(history.history["acc"][-1])

real_end = datetime.now()

print(real_end-real_start)
