import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



whole_dataset = pd.read_csv("german_credt.csv")
X = whole_dataset.drop('Creditability', axis=1)
X = np.asarray(X)
Y = whole_dataset['Creditability']
Y = np.asarray(Y)
Y1= []
for el in Y:
    if el == 0:
        Y1.append(-1)
    else:
        Y1.append(el)
Y = np.asarray(Y1)

X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X,Y,test_size=0.20)

kf = StratifiedKFold(n_splits=5)
sum = 0
train_acc_list = []
test_acc_list = []
print("DECISION TREES")
for train,test in kf.split(X_train1,Y_train1):
    X_train = X_train1[train]
    X_test = X_train1[test]
    Y_train = Y_train1[train]
    Y_test = Y_train1[test]
    # print(train_data.shape)
    # print(test_data.shape)
    # X_train = train_data[:, :-1]
    # Y_train = train_data[:, train_data.shape[1]-1]
    # X_test = test_data[:, :-1]
    # Y_test = test_data[:, test_data.shape[1]-1]
    classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=0.30, min_samples_leaf=0.20)
    classifier.fit(X_train, Y_train)
    #For training data
    train_prediction = classifier.predict(X_train)
    train_acc = accuracy_score(Y_train, train_prediction)
    train_err = 1-train_acc
    test_prediction = classifier.predict(X_test)
    test_acc = accuracy_score(Y_test, test_prediction)
    test_err = 1 - test_acc

    # print('Train acc = ' + str(train_acc) + ' Test acc = ' + str(test_acc))
    # print('Train err = ' + str(train_err) + ' Test err = ' + str(test_err) + '\n')
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

train_variance =  np.var(np.asarray(train_acc_list))
test_variance = np.var(np.asarray(test_acc_list))
y_pred_test = classifier.predict(X_test1)
holdout_test_acc = accuracy_score(Y_test1, y_pred_test)

print('VAR_train_acc = ', train_variance)
print('VAR_test_acc = ', test_variance)
print('Avg_train_acc =' + str(math.fsum(train_acc_list)/len(train_acc_list)))
print('Avg_test_acc = ' + str(math.fsum(test_acc_list)/len(test_acc_list) ))
print("holdout_test_accuracy =", holdout_test_acc)


print("RANDOM FORERSTS")
kf = StratifiedKFold(n_splits=5)
sum = 0
train_acc_list = []
test_acc_list = []
for train,test in kf.split(X_train1,Y_train1):
    X_train = X_train1[train]
    X_test = X_train1[test]
    Y_train = Y_train1[train]
    Y_test = Y_train1[test]
    # print(train_data.shape)
    # print(test_data.shape)
    # X_train = train_data[:, :-1]
    # Y_train = train_data[:, train_data.shape[1]-1]
    # X_test = test_data[:, :-1]
    # Y_test = test_data[:, test_data.shape[1]-1]
    classifier = RandomForestClassifier(n_estimators=1600, max_depth=6, min_samples_split=0.17)
    classifier.fit(X_train, Y_train)
    #For training data
    train_prediction = classifier.predict(X_train)
    train_acc = accuracy_score(Y_train, train_prediction)
    train_err = 1-train_acc
    test_prediction = classifier.predict(X_test)
    test_acc = accuracy_score(Y_test, test_prediction)
    test_err = 1 - test_acc

    # print('Train acc = ' + str(train_acc) + ' Test acc = ' + str(test_acc))
    # print('Train err = ' + str(train_err) + ' Test err = ' + str(test_err) + '\n')
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

train_variance =  np.var(np.asarray(train_acc_list))
test_variance = np.var(np.asarray(test_acc_list))
y_pred_test = classifier.predict(X_test1)
holdout_test_acc = accuracy_score(Y_test1, y_pred_test)

print('Avg_train_acc =' + str(math.fsum(train_acc_list)/len(train_acc_list)))
print('Avg_test_acc = ' + str(math.fsum(test_acc_list)/len(test_acc_list) ))
print('VAR_train_acc = ', train_variance)
print('VAR_test_acc = ', test_variance)
print("holdout_test_accuracy =", holdout_test_acc)