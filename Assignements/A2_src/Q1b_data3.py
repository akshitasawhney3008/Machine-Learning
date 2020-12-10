import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

filename = 'data_3.h5'

def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        x = hf['x'][:]
        y = hf['y'][:]
    return x, y


# def predict(features,w,b):
#     classify = np.sign(np.dot(features, w) + b)
#     return classify

X, Y = load_h5py(filename)

model = SVC(kernel='linear')
model.fit(X,Y)


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
plt.figure(figsize=(15, 10))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.show()



