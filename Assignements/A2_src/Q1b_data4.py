import h5py
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

filename = 'data_4.h5'


# Load the data
def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        x = hf['x'][:]
        y = hf['y'][:]
    return x, y

X, Y = load_h5py(filename)

X[:, 0] = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
X[:, 1] = np.arctan2(X[:, 1], X[:, 0])

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
model = svm.SVC(kernel='linear', C=5.0, gamma='auto', decision_function_shape='ovr', random_state=0, tol=1e-3, max_iter=-1)
model.fit(X, Y)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
plt.figure(figsize=(15, 10))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)


plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
plt.show()
