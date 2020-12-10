import h5py
import matplotlib.pyplot as plt
import math
from sklearn import svm
import numpy as np

filename = 'data_2.h5'


def load_h5py(filename):
    h5 = h5py.File(filename, 'r')
    keylist = h5.keys()
    for key in keylist:
        matrix = h5.get(key)
        matrix = matrix.value
        length = len(matrix.transpose())
        if length == 2:
            x = matrix
        else:
            y = matrix
    return x, y


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_vector = np.vectorize(sigmoid)

X, Y = load_h5py(filename)


X[:, 0] = sigmoid_vector(X[:, 0]**2)
X[:, 1] = X[:, 1]**3

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
model = svm.SVC(kernel='linear', C=6.0, gamma='auto')
model.fit(X, Y)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
plt.figure(figsize=(15, 10))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)


plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
plt.show()