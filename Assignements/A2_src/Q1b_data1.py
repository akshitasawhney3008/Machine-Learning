import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import math

filename = 'data_1.h5'

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        x = hf['x'][:]
        y = hf['y'][:]
    return x, y


# def predict(features,w,b):
#     classify = np.sign(np.dot(features, w) + b)
#     return classify

X, Y = load_h5py(filename)

X[:, 0] = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
# X[:, 1] = np.arctan2(X[:, 1], X[:, 0])
# X[:, 1] = np.log(X[:, 1]**2)
# X[:, 0] = ((X[:, 0]+3)**(2))
# X[:, 0] = ((X[:, 0]+3)*(X[:, 1]+3))
# X[:, 1] = np.log(X[:, 1]+3)
# X[:, 1] = (X[:, 1]+3)**3
# X[:, 1] = X[:, 0]**2 + X[:, 1]**2


model = SVC(kernel='linear')
model.fit(X,Y)


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy  = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
plt.figure(figsize=(15, 10))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy , Z, cmap=plt.cm.coolwarm, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c = Y,cmap=plt.cm.coolwarm)
plt.show()



