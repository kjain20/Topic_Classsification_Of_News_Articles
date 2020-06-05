import numpy as np
from scipy.sparse import csr_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.datasets import fetch_rcv1
#from tensorflow.keras import regularizers
from datetime import datetime
from sklearn.metrics import accuracy_score

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
test_ind = np.random.choice(X_test.shape[0], 50000, replace=False)
X_test = X_test[test_ind]
y_test = y_test[test_ind]

#hidden_nodes = [50,100,150,200,300]
hidden_nodes = [200]
k = 1000
n_iter = 5

final_acc = []
time = []
for h in hidden_nodes:
    start = datetime.now()
    model = Sequential()
    model.add(Dense(h, input_dim=X_train.shape[1], activation='relu',init="random_normal"))
    model.add(Dense(h, activation='relu',init="random_normal"))
#   model.add(Dense(h, activation='relu',init="random_normal"))
#   model.add(Dense(h, activation='relu',init="random_normal"))
#   model.add(Dense(h, activation='relu',init="random_normal"))
    model.add(Dense(1, activation='sigmoid',init="random_normal"))
    sgd=optimizers.SGD(lr=0.1,momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=n_iter, batch_size=k,verbose=1) 
    end = datetime.now()
    time.append(end-start)
    final_acc.append(history.history["acc"][-1])

y_pred = model.predict(X_test,batch_size=k).reshape(X_test.shape[0])
y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0
print(accuracy_score(y_pred,y_test))

real_end = datetime.now()

print(real_end-real_start)
