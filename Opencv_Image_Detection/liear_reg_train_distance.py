import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import linear_model
from numpy.linalg import inv

train_X = np.array([[180/1200, 310/900, 1], [130/1200, 242/900, 1], [142/1200, 231/900, 1], [103/1200, 165/900, 1], [80/1200, 130/1200, 1], [84/1200, 124/900, 1],[85/1200, 130/900, 1], [70/1200, 104/900, 1], [57/1000, 100/1000, 1]])
train_Y = np.array([[2.3], [3.1], [3.8], [4.5], [5.1], [5.3], [5.9], [6.1], [7.3]])
#train_X = np.array([[180/1200, 310/900, 1], [130/1200, 242/900, 1]])
#train_Y = np.array([[2.3], [3.1]])

w1 = 0.1
w2 = 0.1
w3 = 0.1
W = np.array([w1, w2, w3])
Y = train_Y.reshape(-1,1)
X = train_X.reshape(-1,3)
a = np.dot(X.T, X)
b = inv(a)
c = np.dot(b,X.T)
W_result = np.dot(c,Y)
W = W_result
print(W_result)
print(W)

test = np.array([180/1200, 310/900, 1])
result = np.dot(test, W)
print("distance: ", result)



