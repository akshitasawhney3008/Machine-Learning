import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.legend_handler import HandlerLine2D
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler



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
# Xtrain = StandardScaler(Xtrain)

Xtest = test_dataset.drop('Creditability', axis=1)
Xtest = np.asarray(Xtest)[:,1:]
# Xtest = StandardScaler(Xtest)
Ytest = test_dataset['Creditability']
Ytest = list(Ytest)
Ytest1 = []
for el in Ytest:
    if el == 0:
        Ytest1.append(-1)
    else:
        Ytest1.append(el)
Ytest = np.asarray(Ytest1)


def plot(parameter, train_results, test_results, xlabel):
    line1, = plt.plot(parameter, train_results, 'b', label="Train AUC")
    line2, = plt.plot(parameter, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("Accuracy Score")
    plt.xlabel(xlabel)
    plt.show()



#MAX_DEPTH
#The first parameter to tune is max_depth. This indicates how deep the tree can be. The deeper the tree, the more splits it has and it captures more information about the data.
# We fit a decision tree with depths ranging from 1 to 32 and plot the training and test accuracy scores.
def func_max_depth(Xtrain, Ytrain, Xtest, Ytest, classifier):
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        if classifier == 'dt':
            dt = DecisionTreeClassifier(max_depth=max_depth)
        elif classifier == 'rf':
            dt = RandomForestClassifier(max_depth= max_depth)
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
        xlabel = "Max Depth"
    plot(max_depths, train_results, test_results, xlabel)



#MIN_SAMPLE_SPLIT
#represents the minimum number of samples required to split an internal node.
# This can vary between considering at least one sample at each node to considering all of the samples at each node.
# When we increase this parameter, the tree becomes more constrained as it has to consider more samples at each node. Here we will vary the parameter from 10% to 100% of the samples
def func_min_samples_splits(Xtrain, Ytrain, Xtest, Ytest, classifier):
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        if classifier == 'dt':
            dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
        elif classifier == 'rf':
            dt = RandomForestClassifier(min_samples_split=min_samples_split)
        dt.fit(Xtrain, Ytrain)
        train_pred = dt.predict(Xtrain)
        train_accuracy = accuracy_score(Ytrain, train_pred)
        train_results.append(train_accuracy)
        y_pred = dt.predict(Xtest)
        test_accuracy = accuracy_score(Ytest, y_pred)
        test_results.append(test_accuracy)
        xlabel = "Min Sample Split"
    plot(min_samples_splits, train_results, test_results,xlabel)




#MIN_SAMPLE_LEAF
#is The minimum number of samples required to be at a leaf node.
# This parameter is similar to min_samples_splits, however, this describe the minimum number of samples of samples at the leafs, the base of the tree.
def func_min_samples_leafs(Xtrain, Ytrain, Xtest, Ytest, classifier):
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        if classifier == 'dt':
            dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        elif classifier == 'rf':
            dt = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
        dt.fit(Xtrain, Ytrain)
        train_pred = dt.predict(Xtrain)
        train_accuracy = accuracy_score(Ytrain, train_pred)
        train_results.append(train_accuracy)
        y_pred = dt.predict(Xtest)
        test_accuracy = accuracy_score(Ytest, y_pred)
        test_results.append(test_accuracy)
        xlabel = "Min Sample leafs"
    plot(min_samples_leafs, train_results, test_results, xlabel)



#MAX_FEATURES represents the number of features to consider when looking for the best split.
def func_max_features(Xtrain, Ytrain, Xtest, Ytest, classifier):
    max_features = list(range(1,Xtrain.shape[1]))
    train_results = []
    test_results = []
    for max_feature in max_features:
        if classifier == 'dt':
            dt = DecisionTreeClassifier(max_features=max_feature)
        elif classifier == 'rf':
            dt = RandomForestClassifier(max_features=max_feature)
        dt.fit(Xtrain, Ytrain)
        train_pred = dt.predict(Xtrain)
        train_accuracy = accuracy_score(Ytrain, train_pred)
        train_results.append(train_accuracy)
        y_pred = dt.predict(Xtest)
        test_accuracy = accuracy_score(Ytest, y_pred)
        test_results.append(test_accuracy)
        xlabel = "Max Features"
    plot(max_features, train_results, test_results, xlabel)

def func_n_estimators(Xtrain, Ytrain, Xtest, Ytest):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    train_results = []
    test_results = []
    for n_estimator in n_estimators:
        dt = RandomForestClassifier(n_estimators=n_estimator)
        dt.fit(Xtrain, Ytrain)
        train_pred = dt.predict(Xtrain)
        train_accuracy = accuracy_score(Ytrain, train_pred)
        train_results.append(train_accuracy)
        y_pred = dt.predict(Xtest)
        test_accuracy = accuracy_score(Ytest, y_pred)
        test_results.append(test_accuracy)
        xlabel = "N_estimators"
    plot(n_estimators, train_results, test_results, xlabel)

def func_randomized_search(Xtrain, Ytrain, Xtest, Ytest):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(Xtrain, Ytrain)
    print(rf_random.best_params_)
    #{'n_estimators': 800, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 100, 'bootstrap': True}


classifiers = ['dt','rf']
for classifier in classifiers:
    func_max_depth(Xtrain, Ytrain, Xtest, Ytest, classifier)
    func_min_samples_splits(Xtrain, Ytrain, Xtest, Ytest, classifier)
    func_min_samples_leafs(Xtrain, Ytrain, Xtest, Ytest, classifier)
    func_max_features(Xtrain, Ytrain, Xtest, Ytest, classifier)
    if classifier == 'rf':
        # func_randomized_search(Xtrain, Ytrain, Xtest, Ytest)
        func_n_estimators(Xtrain, Ytrain, Xtest, Ytest)


