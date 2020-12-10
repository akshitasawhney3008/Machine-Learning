import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib


def load_file(file_n):
    with h5py.File(file_n, 'r') as hf:
        x = hf['X'][:]
        y = hf['Y'][:]
    return x, y


def make_labels(target_col):
    my_list1 = []
    my_list2 = []
    for i in range(len(target_col)):
        if target_col[i] == 7:
            my_list1.append(0)
        else:
            my_list1.append(1)
        my_list2.append(my_list1)
        my_list1 = []
    return my_list2


X, Y = load_file("MNIST_Subset.h5")
X = X.reshape(X.shape[0],-1)
Xtrain, Xtest, Ytrain , Ytest = train_test_split(X, Y, test_size=0.3)

Ytrain = np.asarray(make_labels(Ytrain))
Ytest = np.asarray(make_labels(Ytest))
Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0



number_nodes_per_layer = [[100],[100,50,50]]
for non in number_nodes_per_layer:
    acc_list = []
    epoch_list = []
    for k in range(400, 410):
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=tuple(non), activation='logistic', max_iter=k)
        clf = clf.fit(Xtrain, Ytrain)
        label_predict = clf.predict_proba(Xtest)

        np_label_predict = np.asarray(label_predict)
        np_label_predict = np_label_predict.astype('float64')

        maximum_probability = -100
        max_prob_idx_list = []
        idx = 0

        for i in range(0, np_label_predict.shape[0]):
            for j in range(0, np_label_predict.shape[1]):
                if np_label_predict[i][j] > maximum_probability:
                    maximum_probability = np_label_predict[i][j]
                    idx = j
            maximum_probability = -100
            max_prob_idx_list.append(idx)

        acc_list.append(accuracy_score(max_prob_idx_list, Ytest)*100)
        epoch_list.append(k)

    max_acc = 0
    for a in acc_list:
        if a > max_acc:
            max_acc = a
    print('Maximum accuracy ' + str(max_acc))
    epoch = acc_list.index(max_acc) + 400
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=tuple(non), activation='logistic', max_iter=epoch)
    clf = clf.fit(Xtrain, Ytrain)
    joblib.dump(clf, 'Objective_1_c.pkl')
    plt.legend(('Test accuracy-1 hidden layer', 'Test accuracy-3 hidden layer'),
               loc='upper right')
    plt.plot(epoch_list, acc_list)
    plt.title('Objective' + str(i))
    plt.xlabel('Epochs' + str(i))
    plt.ylabel('Accuracy' + str(i))
    # plt.savefig('Objective_1c' + str(i))
    plt.show()
