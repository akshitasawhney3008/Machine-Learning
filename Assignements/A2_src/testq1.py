import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import random
from numpy import dot
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy.special import expit

learning_rate = 0.01


def load_file(file_n):
    with h5py.File(file_n, 'r') as hf:
        x = hf['X'][:]
        y = hf['Y'][:]
    return x, y


def initialize_weights(number_of_layers,number_of_nodes_per_layer,input_data):
    weights = []
    for i in range(number_of_layers+1):
        if i == 0:
            weights.append(2 * random.random((input_data.shape[1], number_of_nodes_per_layer[i])) - 1)
        elif i == number_of_layers:
            weights.append(2 * random.random((number_of_nodes_per_layer[i-1], 1)) - 1)
        else:
            weights.append(2 * random.random((number_of_nodes_per_layer[i-1], number_of_nodes_per_layer[i])) - 1)
    return weights


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    for i in range(0, len(x)):
        for k in range(len(x[i])):
            if x[i][k] > 0:
                x[i][k] = 1
            else:
                x[i][k] = 0
    return x


def softmax(x):
    expA = expit(x - np.max(x))
    return expA / expA.sum(0)



# def softmax_grad(softmax):
#     s = softmax.reshape(-1,1)
#     return np.diagflat(s) - np.dot(s, s.T)


def forward_propagation(weights,input,number_of_layers):
    layer = []
    for i in range(number_of_layers+1):
        if i == number_of_layers:
            hidden_layer = layer[i - 1]
            dot_prod = dot(hidden_layer, weights[i])
            layer.append(softmax(dot_prod))
        elif i == 0:
            w = weights[i]
            dot_prod = dot(input, w)
            layer.append(relu(dot_prod))
        else:
            hidden_layer = layer[i-1]
            dot_prod = dot(hidden_layer, weights[i])
            layer.append(relu(dot_prod))
    return layer


def backward_propagation(output,layer,number_of_layers,weights,input):
    for i in range(number_of_layers,-1,-1):
        if i == number_of_layers:
            output_layer = layer[- 1]
            delta = output - output_layer
            # delta = outputError * softmax_grad(softmax(output_layer))
            out_weights_adjustment = learning_rate * dot(layer[i-1].T, delta)
            weights[i] += out_weights_adjustment
        elif i == 0:
            delta = dot(delta, weights[i+1].T) * relu_derivative(layer[i])
            weight_1_adjustment = learning_rate * dot(input.T, delta)
            weights[i] += weight_1_adjustment
        else:
            delta = dot(delta, weights[i+1].T) * relu_derivative(layer[i])
            weight_adjustment = learning_rate * dot(layer[i-1].T, delta)
            weights[i] += weight_adjustment
    return weights


def train(input_data, output_data, number_of_layers, no_of_steps, weights):
    for j in range(no_of_steps):
        layer = forward_propagation(weights,input_data,number_of_layers)
        weights = backward_propagation(output_data,layer,number_of_layers,weights,input_data)
    return weights


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


def predict(x, weights):
    layer_pred = []
    for i in range(len(weights)):
        if i == number_of_layers:
            layer_pred.append(softmax(dot(layer_pred[i - 1], weights[i])))
        elif i == 0:
            layer_pred.append((relu(dot(x, weights[i]))))
        else:
            layer_pred.append(relu(dot(layer_pred[i-1], weights[i])))

    return layer_pred[-1]


def accuracy(predicted_y, actualy_y):
    counter = 0.0
    for i in range(0, len(actualy_y)):
        if predicted_y[i] == actualy_y[i]:
            counter += 1
    my_accuracy= counter / len(predicted_y) * 100
    return my_accuracy


X, Y = load_file("MNIST_Subset.h5")
X = X.reshape(X.shape[0],-1)
Xtrain, Xtest, Ytrain , Ytest = train_test_split(X, Y, test_size=0.3)

Xtrain = Xtrain/255
Xtest = Xtest/255
my_list2 = make_labels(Ytrain)

input_data = Xtrain
output_data = my_list2



number_of_layers = [1]
number_nodes_per_layer = [[100]]
weights = []
for i in range(len(number_of_layers)):
    my_epoch = []
    my_acc = []
    my_acc_train = []
    weights = initialize_weights(number_of_layers[i],number_nodes_per_layer[i],input_data)
    for j in range(400,500):
        my_epoch.append(j)
        # for k in range(len(Xtrain)):
        #     my_list.append(predict(Xtrain[k],weights))
        # accuracy = accuracy(Xtrain, Ytrain)
        # print(accuracy)
        my_nn = train(input_data,output_data,number_of_layers[i],j,weights)

        my_list_train = []
        for k in range(len(Xtrain)):
            my_list_train.append(predict(Xtrain[k], my_nn))

        my_list = []
        for k in range(len(Xtest)):
            my_list.append(predict(Xtest[k],my_nn))

        my_list3 = make_labels(Ytest)
        my_list3 = sum(my_list3, [])

        my_list4 = make_labels(Ytrain)
        my_list4 = sum(my_list4, [])

        for k in range(len(my_list_train)):
            if my_list_train[k] > 0.5:
                my_list_train[k] = 1
            else:
                my_list_train[k] = 0

        for k in range(len(my_list)):
            if my_list[k] > 0.5:
                my_list[k] = 1
            else:
                my_list[k] = 0

        counter = 0.0
        my_acc_train.append(accuracy(my_list_train, my_list4))
        my_acc.append(accuracy(my_list, my_list3))

    max_accuracy = 0
    for a in my_acc:
        if a > max_accuracy:
            max_accuracy = a
    par = 400 + my_acc.index(max_accuracy)
    print("max_accuracy:" , max_accuracy)
    print(par)
    my_nn = train(input_data,output_data,number_of_layers[i],par,weights)
    joblib.dump(my_nn, 'Objective_1b.pkl')
    plt.plot(my_epoch, my_acc)
    plt.title('Objective' + str(i))
    plt.xlabel('Epochs'+ str(i))
    plt.ylabel('Accuracy' + str(i))
    plt.savefig('Objective(1b)' + str(i))


