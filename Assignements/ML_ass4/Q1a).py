import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
import math


train_dataset = pd.read_csv("Test_Dataset.csv")
test_dataset = pd.read_csv("Training_Dataset.csv")

Xtrain = train_dataset.drop('Creditability', axis=1)
Xtrain = np.asarray(Xtrain)[:,1:]
Ytrain = train_dataset['Creditability']
Ytrain = list(Ytrain)
Ytrain1 = []
for el in Ytrain:
    if el == 0:
        Ytrain1.append(-1)
    else:
        Ytrain1.append(el)
Ytrain = np.asarray(Ytrain1)
# Xtrain = normalize(Xtrain)

Xtest = test_dataset.drop('Creditability', axis=1)
Xtest = np.asarray(Xtest)[:,1:]
# Xtest = normalize(Xtest)
Ytest = test_dataset['Creditability']
Ytest = list(Ytest)
Ytest1 = []
for el in Ytest:
    if el == 0:
        Ytest1.append(-1)
    else:
        Ytest1.append(el)
Ytest = np.asarray(Ytest1)
#
# classifier = DecisionTreeClassifier(max_depth=10,min_samples_split=0.40, min_samples_leaf=0.20)
# # classifier.fit(Xtrain, Ytrain)
kf = StratifiedKFold(n_splits=10)
sum = 0
train_acc_list = []
test_acc_list = []
for train,test in kf.split(Xtrain,Ytrain):
    X_train = Xtrain[train]
    X_test = Xtrain[test]
    Y_train = Ytrain[train]
    Y_test = Ytrain[test]
    # print(train_data.shape)
    # print(test_data.shape)
    # X_train = train_data[:, :-1]
    # Y_train = train_data[:, train_data.shape[1]-1]
    # X_test = test_data[:, :-1]
    # Y_test = test_data[:, test_data.shape[1]-1]
    classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=0.40, min_samples_leaf=0.20)
    classifier.fit(X_train, Y_train)
    #For training data
    train_prediction = classifier.predict(X_train)
    train_acc = accuracy_score(Y_train, train_prediction)
    train_err = 1-train_acc
    test_prediction = classifier.predict(X_test)
    test_acc = accuracy_score(Y_test, test_prediction)
    test_err = 1 - test_acc

    # print('Train acc = ' + str(train_acc) + ' Test acc = ' + str(test_acc) + '\n')
    # print('Train err = ' + str(train_err) + ' Test err = ' + str(test_err) + '\n')
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


# y_pred_train = classifier.predict(Xtrain)
y_pred_test = classifier.predict(Xtest)


# train_acc = accuracy_score(Ytrain,y_pred_train)
holdout_test_acc = accuracy_score(Ytest, y_pred_test)

# print(train_acc)

# holdout_test_prediction = mysvclassifier.predict(whole_dataset_test)
# holdout_test_prediction = mysvclassifier.predict(X_test1)
# # holdout_test_acc = accuracy_score(targetcol_test, holdout_test_prediction)
# holdout_test_acc = accuracy_score(Y_test1, holdout_test_prediction)
print("DECISION TREE")
print('Avg_train_acc =' + str(math.fsum(train_acc_list)/len(train_acc_list) *100))
print('Avg_test_acc = ' + str(math.fsum(test_acc_list)/len(test_acc_list)*100 ))
print("holdout_test_accuracy =", holdout_test_acc * 100)
joblib.dump(classifier, 'DecisionTrees.pkl')

#RANDOM_FOREST

kf = StratifiedKFold(n_splits=10)
sum = 0
train_acc_list = []
test_acc_list = []
for train,test in kf.split(Xtrain,Ytrain):
    X_train = Xtrain[train]
    X_test = Xtrain[test]
    Y_train = Ytrain[train]
    Y_test = Ytrain[test]
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
    #
    # print('Train acc = ' + str(train_acc) + ' Test acc = ' + str(test_acc) + '\n')
    # # print('Train err = ' + str(train_err) + ' Test err = ' + str(test_err) + '\n')
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


# y_pred_train = classifier.predict(Xtrain)
y_pred_test = classifier.predict(Xtest)


# train_acc = accuracy_score(Ytrain,y_pred_train)
holdout_test_acc = accuracy_score(Ytest, y_pred_test)

# print(train_acc)

# holdout_test_prediction = mysvclassifier.predict(whole_dataset_test)
# holdout_test_prediction = mysvclassifier.predict(X_test1)
# # holdout_test_acc = accuracy_score(targetcol_test, holdout_test_prediction)
# holdout_test_acc = accuracy_score(Y_test1, holdout_test_prediction)
print("\nRANDOM FOREST")
print('Avg_train_acc =' + str(math.fsum(train_acc_list)/len(train_acc_list) * 100))
print('Avg_test_acc = ' + str(math.fsum(test_acc_list)/len(test_acc_list) * 100))
print("holdout_test_accuracy =", holdout_test_acc*100)
joblib.dump(classifier, 'RandomForest.pkl')

