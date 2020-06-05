#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_rcv1
import numpy as np
from datetime import datetime 
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
import math
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

start = datetime.now()

rcv1 = fetch_rcv1()

data = csr_matrix(rcv1["data"])
target = rcv1["target"].toarray()

# CREATING LABELS
labels = np.add(np.zeros((target.shape[0])),-1)
labels[np.where(target[:,33]==1)[0]] += 2

X_train = data[:100000]
y_train = labels[:100000]
X_test = data[100000:]
y_test = labels[100000:]


# ADAGRAD ALGORITHM

n_iter = 200  # Number of Iterations
l = 0.001  # Regularization Coefficient
#batch_size_list = [50,100,500,1000,5000,10000] 
batch_size_list = [10000] # Number of Training instances chosen during one iteration
w = csr_matrix(np.divide(np.random.rand(1,data.shape[1]),10)) # Weights
g_total = np.zeros(w.shape)

x_axis = np.add(range(n_iter),1)

for batch_size in batch_size_list:
    accuracy_list = []
    t = 0
    for i in range(n_iter):
        t += 1
        print(t)
        A_t_ind = np.random.choice(X_train.shape[0],batch_size,replace=False)
        A_t = X_train[A_t_ind]
        y_t = y_train[A_t_ind].reshape((batch_size,1))
        learning_coef = 1.0/(t**0.5)
        y_pred = np.dot(w,np.transpose(A_t)).toarray().reshape((batch_size,1))
        y_pred[y_pred>=0] = 1
        y_pred[y_pred<0] = -1
        y_y_pred = np.multiply(y_t,y_pred)
        A_plus_ind = np.where(y_y_pred < 1)[0]
        grad_loss = (w.multiply(learning_coef*l) - csr_matrix(((csr_matrix(A_t)[A_plus_ind]).multiply(csr_matrix(y_t)[A_plus_ind]).sum(axis=0))).multiply(learning_coef/batch_size)).toarray()        
        g_total += np.square(grad_loss)
        G = np.sqrt(g_total)
        G_inv = 1.0/G
        w_1_2 = w - csr_matrix(np.multiply(G_inv,grad_loss)).multiply(learning_coef)
        min_comp = (1/l**0.5)*(1.0/np.linalg.norm(np.multiply(w_1_2.toarray(),G),ord=2))
        temp_min = np.minimum(1,min_comp)
        w = w_1_2.multiply(temp_min)
        y_train_pred = w.dot(np.transpose(X_train)).toarray().reshape(X_train.shape[0])
        y_train_pred[y_train_pred>=0] = 1
        y_train_pred[y_train_pred<0] = -1
        accuracy_list.append(1.0-accuracy_score(y_train,y_train_pred))
    plt.plot(x_axis,accuracy_list,label="Batch Size = " + str(batch_size))

# TEST ACCURACY
y_pred = w.dot(np.transpose(X_test)).toarray().reshape(X_test.shape[0])
y_pred[y_pred>=0] = 1
y_pred[y_pred<0] = -1
y_test = [a for a in y_test]
y_pred = [a for a in y_pred]
print(accuracy_score(y_test,y_pred))

plt.plot(x_axis,accuracy_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Training Error")
plt.title("ADAGRAD")
plt.legend()

end = datetime.now()
print(end-start)




  
  
  
  
  
  
  
  
  