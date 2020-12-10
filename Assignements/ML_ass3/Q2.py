import pickle
import torch
import torch.nn as nn
from torchvision import models,transforms,datasets
from torch.autograd import Variable
import tensorflow as tf
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import math


def load(file_name, img_transform):
    data_path = file_name
    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=img_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader


def load_file(filename):
    infile = open(filename, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict


imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))])

train = load_file("train_CIFAR.pickle")
for key in train.keys():
    if key == 'X':
        Xtrain = train[key]
    else:
        Ytrain = train[key]

test = load_file("test_CIFAR.pickle")
for key in test.keys():
    if key == 'X':
        Xtest = test[key]
    else:
        Ytest = test[key]


wholedata = np.append(Xtrain,Xtest,axis=0)
target = np.append(Ytrain,Ytest)

model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
output = model((torch.from_numpy(wholedata)))
print("hi")


X_train1,X_test1,Y_train1,Y_test1 = train_test_split(output, target, test_size=0.30)

print('Training model/n')
C_range = np.logspace(-2, 1, 20)
param_grid = dict(C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
grid = GridSearchCV(LinearSVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train1, Y_train1)

print("The best parameters are %s with a sore of %0.2f"
      % (grid.best_params_, grid.best_score_))
for key,values in grid.best_params_.items():
    if key == 'C':
        C = values


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
    mysvcclassifier = mysvclassifier = LinearSVC(C=C)
    mysvclassifier = mysvclassifier.fit(X_train, Y_train)
    #For training data
    train_prediction = mysvclassifier.predict(X_train)
    train_acc = accuracy_score(Y_train, train_prediction)
    train_err = 1-train_acc
    test_prediction = mysvclassifier.predict(X_test)
    test_acc = accuracy_score(Y_test, test_prediction)
    test_err = 1 - test_acc

    # print('Train acc = ' + str(train_acc) + ' Test acc = ' + str(test_acc) + '\n')
    # print('Train err = ' + str(train_err) + ' Test err = ' + str(test_err) + '\n')
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(len(test_acc_list))
    print(train_acc)
    print(test_acc)

holdout_test_prediction = mysvclassifier.predict(X_test1)
# holdout_test_acc = accuracy_score(targetcol_test, holdout_test_prediction)
holdout_test_acc = accuracy_score(Y_test1, holdout_test_prediction)
print('Avg_train_acc =' + str(math.fsum(train_acc_list)/len(train_acc_list)))
print('Avg_test_acc = ' + str(math.fsum(test_acc_list)/len(test_acc_list) ))
print("holdout_test_accuracy =" , holdout_test_acc)
