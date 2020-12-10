import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import random

filename = 'data_4.h5'


def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        x = hf['x'][:]
        y = hf['y'][:]
    return x, y

def split(whole_data, folds):
    all_splits = []
    temp_data = list(whole_data)
    size_of_fold = int(len(temp_data)*folds)
    split_data = []
    while(len(split_data)< size_of_fold):
        ind = random.randrange(len(temp_data))
        split_data.append(temp_data.pop(ind))
    split_data = np.asarray(split_data)
    test_data = np.asarray(temp_data)
    return split_data,test_data

def transform_data(train_data,i):
    new_train_label = []
    temp_data = list(train_data)
    for k in range(len(train_data)):
        if(train_data[k][train_data.shape[1]-1] == i):
            new_train_label.append(1)
        else:
            new_train_label.append(-1)
    train_label = np.asarray(new_train_label)
    train_data = train_data[:, :-1]
    return train_data,train_label


def predict(features,w,b):
    z = np.dot(features, w)
    z = z + b
    classify = 1/(1+(np.exp(-z)))
    return classify

def predict_class(features,w,b):
    classify = np.sign(np.dot(features, w) + b)
    return classify


def perf_measure(y_actual,y_pred, k ,j):
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == k:
            TP += 1
        if y_pred[i] == k and y_actual[i] != y_pred[i]:
            FP += 1

        if y_pred[i] == j and y_actual[i] != y_pred[i]:
            FN += 1

    return (TP, FP, FN)

def f1_score(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1

X, Y = load_h5py(filename)

wholedata = np.c_[X, Y]
train_data,test_data = split(wholedata,0.8)
number_of_classes = 2
classify_list = []
for i in range(number_of_classes):
    transformed_train_label = []
    transformed_test_label = []
    new_train_data, new_train_label = transform_data(train_data, i)
    classifier = SVC(kernel='linear')
    classifier.fit(new_train_data, new_train_label)
    w = classifier.coef_
    b = classifier.intercept_
    classify = predict(test_data[:, :-1], w.transpose(), b)
    classify = classify.tolist()
    classify = sum(classify, [])

    classify_class = predict_class(test_data[:, :-1], w.transpose(), b)
    classify_class = list(classify_class)

    t_classify = []
    for cl in classify_class:
        if cl == 1:
            t_classify.append(i)
        else:
            t_classify.append(1)

    t_actual = test_data[:, -1]
    TP, FP, FN = perf_measure(test_data[:, -1], t_classify, i, 1)
    F1 = f1_score(TP, FP, FN)
    print(F1)
    classify_list.append(classify)
classify_arr = np.asarray(classify_list)
final_classification = []
for sample in classify_arr.transpose():
    sample = list(sample)
    final_classification.append(sample.index(max(sample)))
test_label = test_data[:,-1]
correct = 0
for i in range(len(final_classification)):
    if final_classification[i] == test_label[i]:
        correct = correct+1
accuracy = (correct/len(final_classification)) * 100
print(accuracy)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
plt.figure(figsize=(15, 10))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.show()


