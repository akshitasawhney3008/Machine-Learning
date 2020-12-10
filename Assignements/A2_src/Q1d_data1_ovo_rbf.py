import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import math
from sklearn.externals import joblib
import random
from sklearn.model_selection import GridSearchCV

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

def transform_data(train_data,i,j):
    new_train_data = []
    temp_data = list(train_data)
    for k in range(len(train_data)):
        if(train_data[k][train_data.shape[1]-1] == i or train_data[k][train_data.shape[1]-1] == j ):
            new_train_data.append(temp_data[k])
    train_data = np.asarray(new_train_data)
    train_label = train_data[:,train_data.shape[1]-1]
    train_data = train_data[:, :-1]
    return train_data,train_label


# def predict(features,w,b):
#     classify = np.sign(np.dot(features, w) + b)
#     return classify

def perf_measure(y_actual,y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1

    return (TP, FP, TN, FN)

def f1_score(TP, FP, TN, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1

X, Y = load_h5py(filename)

wholedata = np.c_[X, Y]
train_data,test_data = split(wholedata,0.8)
classify_list = []
number_of_classes = 2

for i in range(number_of_classes-1):
    for j in range(i+1,number_of_classes):
        transformed_train_label = []
        transformed_test_label = []
        new_train_data , new_train_label = transform_data(train_data,i,j)
        for el in new_train_label:
            if el == i:
                transformed_train_label.append(1)
            else:
                transformed_train_label.append(-1)
        new_train_label = np.asarray(transformed_train_label)
        # C_range = np.logspace(-1, 10, 10)
        # # gamma_range = np.logspace(-10, 3, 2)
        # param_grid = dict(C=C_range)
        #
        # # param_grid = dict(C=C_range)
        # gs = GridSearchCV(SVC(), param_grid=param_grid)
        # gs.fit(new_train_data,new_train_label)
        #
        # print('The parameters combination that would give best accuracy is : ')
        # print(gs.best_params_)
        # print('The best accuracy achieved after parameter tuning via grid search is : ', gs.best_score_)
        # best_params = gs.best_params_
        # for key, val in best_params.items():
        #     if key == 'C':
        #         C = best_params[key]
        #     else:
        #         gamma = best_params[key]

        sum = 0

        classifier = SVC(C=1.1)
        # joblib.dump(classifier, 'mysvcclassifier.pkl')

        classifier.fit(new_train_data,new_train_label)
        # w = classifier.coef_
        # b = classifier.intercept_
        # new_test_data,new_test_label = transform_data(test_data,i,j)
        # for el in new_test_label:
        #     if el == i:
        #         transformed_test_label.append(1)
        #     else:
        #         transformed_test_label.append(-1)
        classify = classifier.predict(test_data[:,:-1])
        classify = list(classify)

        t_classify = []
        for cl in classify:
            if cl == 1:
                t_classify.append(i)
            else:
                t_classify.append(j)
        classify = np.asarray(t_classify)
        classify_list.append(classify)
        TP, FP, TN, FN = perf_measure(test_data[:,-1], classify)
        F1 = f1_score(TP, FP, TN, FN)
        print(F1)
classify_arr = np.asarray(classify_list)
classify_arr = classify_arr.transpose()
final_classification = []
for row in classify_arr:
    unique, counts = np.unique(row, return_counts=True)
    dict_cout = dict(zip(unique, counts))
    value1 = 0
    for key,value in dict_cout.items():
        if value > value1:
            value1 = value
            key1 = key
    final_classification.append(key1)
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
