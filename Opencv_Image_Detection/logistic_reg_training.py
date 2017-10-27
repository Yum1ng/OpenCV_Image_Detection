import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import linear_model
# valid_X = np.array(sio.loadmat("Valid_X.mat")['Valid_X'])
train_X = np.array(sio.loadmat("Train_X.mat")['Train_X'])
train_Y = np.array(sio.loadmat("Train_Y.mat")['Train_Y'])

# valid_Y = np.array(sio.loadmat("Valid_Y.mat")['Valid_Y'])
'''
logreg = linear_model.LogisticRegression()
train_X.resize(40*900*1200, 3)
train_Y.resize(40*900*1200, 1)
print("train_x shape: ", train_X.shape)
print("train_y shape: ", train_Y.shape)
logreg.fit(train_X, train_Y)
print("params: ", logreg.get_params())
'''
def sigmoid(t):
    return 1/(1+np.exp(-1*t))

w1 = 2.83041518
w2 = 7.28183566
w3 = -0.49499156
w4 = -6.02723031
W = np.array([w1, w2, w3, w4])
alpha = 0.000004
error = 0
iterations = 4
error_rate = np.array(np.zeros(15))
numbers = np.arange(15)
barrel_corect_rate = np.array(np.zeros(15))
wrong_barrel_rate =  np.array(np.zeros(15))
totalerror = np.array(np.zeros(iterations))
for t in range(iterations):
    for k in range(15):
        error = 0
        dW = np.array([0.0, 0.0, 0.0, 0.0])
        print("k is: ", k)
        wrong_barrel = 0
        barrel_correct = 0
        barrel_total = 0
        count = 0
        for i in range(train_X.shape[1]):

            for j in range(train_X.shape[2]):
                X = np.array(train_X[k, i, j, :])
                xb = np.array(1)
                X = np.hstack((X, xb))
                y = train_Y[k, i, j]
                g = np.dot(X, W.transpose())
                if g > 10 or g < -10:
                    if i % 500 == 0 and j % 500 == 0:
                        print("i is : ", i, " j : ", j, " x0 : ", X[0], "x1: ", X[1], "x2: ", X[2], "x3: ", X[3], " g is: ", g)
                y_result = sigmoid(g)
                if y_result > 0.5:
                    count += 1
                if y == 1:
                    barrel_total += 1
                    if y_result > 0.5:
                        barrel_correct += 1
                if (y_result > 0.5 and y == 0) or (y_result < 0.5 and y == 1):
                    error += 1
                    totalerror[t] += 1
                    if y_result < 0.5:
                        dW += (y_result - y) * X * 10 # increase the penalty when a red barrel is wrong detected
                    if y_result > 0.5:
                        wrong_barrel += 1
                        dW += (y_result - y) * X * 10
                dW += (y_result - y)*X
        wrong_barrel_rate[k] = wrong_barrel/barrel_total
        error_rate[k] = error/(train_X.shape[1]*train_X.shape[2])
        barrel_corect_rate[k] = barrel_correct/barrel_total
        print("w is : ", W, " dw is: ", dW, "error rate is ", error_rate[k], "barrel corect rate is : ", barrel_corect_rate[k])
        print("wrong_barrel_rate: ", wrong_barrel_rate[k])
        print("y_result > 0.5 count: ", count)
        W -= alpha * dW
    print("total error rate: ", totalerror[t]/(15*train_X.shape[1]*train_X.shape[2]))
print("end")
plt.plot(numbers, error_rate)
plt.xlabel("Iteration")
plt.ylabel("Training Error")
plt.title("Training")

plt.show()
plt.hold(True)
plt.plot(numbers, barrel_corect_rate)
plt.xlabel("Iteration")
plt.ylabel("Barrel correct rate")
plt.title("Training Barrel correct rate")
plt.show()








