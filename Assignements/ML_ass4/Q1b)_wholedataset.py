import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


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

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X ,Y ,test_size=0.20)

#MAX_DEPTH
#The first parameter to tune is max_depth. This indicates how deep the tree can be. The deeper the tree, the more splits it has and it captures more information about the data.
# We fit a decision tree with depths ranging from 1 to 32 and plot the training and test accuracy scores.
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(Xtrain, Ytrain)
   train_pred = dt.predict(Xtrain)
   train_accuracy = accuracy_score(Ytrain, train_pred)
   # roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add acc score to previous train results
   train_results.append(train_accuracy)
   y_pred = dt.predict(Xtest)
   test_accuracy = accuracy_score(Ytest, y_pred)
   # roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add acc score to previous test results
   test_results.append(test_accuracy)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("Accuracy Score")
plt.xlabel("Tree depth")
plt.show()

#MIN_SAMPLE_SPLIT
#represents the minimum number of samples required to split an internal node.
# This can vary between considering at least one sample at each node to considering all of the samples at each node.
# When we increase this parameter, the tree becomes more constrained as it has to consider more samples at each node. Here we will vary the parameter from 10% to 100% of the samples

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
   dt.fit(Xtrain, Ytrain)
   train_pred = dt.predict(Xtrain)
   train_accuracy = accuracy_score(Ytrain, train_pred)
   train_results.append(train_accuracy)
   y_pred = dt.predict(Xtest)
   test_accuracy = accuracy_score(Ytest, y_pred)
   test_results.append(test_accuracy)

line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("Accuracy Score")
plt.xlabel("MIN_SAMPLE_SPLIT")
plt.show()


#MIN_SAMPLE_LEAF
#is The minimum number of samples required to be at a leaf node.
# This parameter is similar to min_samples_splits, however, this describe the minimum number of samples of samples at the leafs, the base of the tree.

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
   dt.fit(Xtrain, Ytrain)
   train_pred = dt.predict(Xtrain)
   train_accuracy = accuracy_score(Ytrain, train_pred)
   train_results.append(train_accuracy)
   y_pred = dt.predict(Xtest)
   test_accuracy = accuracy_score(Ytest, y_pred)
   test_results.append(test_accuracy)
line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel("Accuracy Score")
plt.xlabel("Min_Sample_Leaf")
plt.show()

#MAX_FEATURES represents the number of features to consider when looking for the best split.

max_features = list(range(1,Xtrain.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
   dt = DecisionTreeClassifier(max_features=max_feature)
   dt.fit(Xtrain, Ytrain)
   train_pred = dt.predict(Xtrain)
   train_accuracy = accuracy_score(Ytrain, train_pred)
   train_results.append(train_accuracy)
   y_pred = dt.predict(Xtest)
   test_accuracy = accuracy_score(Ytest, y_pred)
   test_results.append(test_accuracy)
line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("Accuracy Score")
plt.xlabel("Max Features")
plt.show()