import h5py
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import random
from numpy import dot
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy.special import expit


learning_rate = 0.0001


def load_file(file_n):
    with h5py.File(file_n, 'r') as hf:
        x = hf['X'][:]
        y = hf['Y'][:]
        x = np.array([i.flatten() for i in x])
        y = (y == 7).astype(int)
    return x, y


def initialize_weights(number_of_layers,number_of_nodes_per_layer,input_data):
    weights = []
    for i in range(number_of_layers + 1):
        if i == 0:
            rng = np.random.uniform(-1.0, 1.0, (number_of_nodes_per_layer[i], input_data.shape[0]))
            weights.append(rng)
        elif i == number_of_layers:
            weights.append(np.random.uniform(-1.0, 1.0,(2,number_of_nodes_per_layer[i - 1])))
        else:
            weights.append(np.random.uniform(-1.0, 1.0, (number_of_nodes_per_layer[i], number_of_nodes_per_layer[i - 1])))
    return weights

# np.random.uniform(-1.0, 1.0, (self.layer_sizes[i], self.layer_sizes[i - 1]))
def sigmoid(x):
    y = 1/(1+expit(-x))
    return y


def sigmoid_derivative(x):
    y = sigmoid(x) * sigmoid(1-x)
    return y


def softmax(x):
    expA = expit(x-np.max(x))
    return expA / expA.sum(axis=1, keepdims=True)


def loss_function(Y, A):
    return Y*np.log(A) + (1.0 - Y)*np.log(1.0 - A)

def loss_derivative_function(Y, A):
    return ((1.0 - Y)/(1.0 - A) - (Y/A))

#
# def softmax_grad(s):
#     jacobian_m = np.diag(s)
#     for i in range(len(jacobian_m)):
#         for j in range(len(jacobian_m)):
#             if i == j:
#                 jacobian_m[i][j] = s[i] * (1 - s[i])
#             else:
#                 jacobian_m[i][j] = -s[i] * s[j]
#     return jacobian_m


def forward_propagation(weights,input_data,number_of_layers):
    layer = []
    for i in range(number_of_layers+1):
        if i == number_of_layers:
            hidden_layer = layer[i - 1]
            dot_prod = dot(weights[i],hidden_layer)
            layer.append(sigmoid(dot_prod))
        elif i == 0:
            w = weights[i]
            dot_prod = dot(w, input_data)
            layer.append(sigmoid(dot_prod))
        else:
            hidden_layer = layer[i-1]
            dot_prod = dot(weights[i],hidden_layer)
            layer.append(sigmoid(dot_prod))
    return layer


def backward_propagation(output_data,layer,number_of_layers,weights,input_data,loss_derivative):
    for i in range(number_of_layers,-1,-1):
        if i == number_of_layers:
            output_layer = layer[- 1]
            delta = output_data - output_layer
            b = dot(weights[i],layer[i-1])
            delta = loss_derivative * sigmoid_derivative(b)
            # delta = outputError * softmax_grad(softmax(output_layer))
            out_weights_adjustment = learning_rate * dot(delta,layer[i-1].T)
            weights[i] += out_weights_adjustment
            # B[i] = B[i] - learning_rate * dB[i]
        elif i == 0:
            delta = dot(weights[i+1].T,delta) * sigmoid_derivative(dot(weights[i],input_data))
            weight_1_adjustment = learning_rate * dot(delta,input_data.T)
            weights[i] += weight_1_adjustment
        else:
            delta = dot(weights[i+1].T,delta) * sigmoid_derivative(dot(weights[i], layer[i-1]))
            weight_adjustment = learning_rate * dot(delta,layer[i-1].T)
            weights[i] += weight_adjustment
    return weights


# def loss_function(Y, A):
#     return Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A)
#
#
# def loss_derivative_function(Y, A):
#     return (1.0 - Y) / (1.0 - A) - (Y / A)


def train(input_data, output_data, number_of_layers, no_of_steps, weights):
    # dA = {}
    for j in range(no_of_steps):
        layer = forward_propagation(weights,input_data,number_of_layers)
        # loss = loss_function(Y, layer[number_of_layers - 1])
        loss_derivative = loss_derivative_function(np.asarray(output_data), layer[number_of_layers])
        # dA[number_of_layers-1] = loss_derivative
        weights = backward_propagation(output_data,layer,number_of_layers,weights,input_data,loss_derivative)
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


def predict(x, weights,number_of_layers):
    # x = x.reshape((x.shape[0],1))
    layer_pred = []
    for i in range(len(weights)):
        if i == number_of_layers:
            layer_pred.append(sigmoid(dot(layer_pred[i-1],weights[i])))
        elif i == 0:
            layer_pred.append((sigmoid(dot(x, weights[i]))))
        else:
            layer_pred.append(sigmoid(dot(layer_pred[i-1],weights[i])))

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
Xtrain, Xtest, Ytrain , Ytest = train_test_split(X, Y, test_size=0.15)


Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0
# my_list2 = make_labels(Ytrain)
input_data = Xtrain
# output_data = my_list2

#Converting y_train to hot encode
b = np.zeros((len(Ytrain), 2))
b[np.arange(len(Ytrain)), Ytrain] = 1
Ytrain = b
Ytrain = np.transpose(Ytrain)
output_data = Ytrain

#Converting y_train to hot encode
b = np.zeros((len(Ytest), 2))
b[np.arange(len(Ytest)), Ytest] = 1
Ytest = b
Ytest = np.transpose(Ytest)
Xtrain,Xtest = np.transpose(Xtrain),np.transpose(Xtest)
input_data = Xtrain

number_of_layers = [3]
number_nodes_per_layer = [[100,50,50]]
weights = []
for i in range(len(number_of_layers)):
    my_epoch = []
    my_acc = []
    my_acc_train = []
    my_weights = []
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
            my_list_train.append(predict(Xtrain[k], my_nn, number_of_layers))

        my_list = []
        for k in range(len(Xtest)):
            my_list.append(predict(Xtest[k],my_nn, number_of_layers))

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
        my_weights.append(my_nn)
    max_accuracy = 0
    for a in my_acc:
        if a > max_accuracy:
            max_accuracy = a
    par = 400 + my_acc.index(max_accuracy)
    print("max_accuracy:", max_accuracy)
    print(par)
    my_nn = train(input_data,output_data,number_of_layers[i],par,my_weights[my_acc.index(max_accuracy)])
    joblib.dump(my_nn, 'Objective_1.pkl')
    plt.plot(my_epoch, my_acc_train)
    plt.plot(my_epoch, my_acc)
    plt.legend(('Train Accuracy', 'Test accuracy'),
               loc='upper right')
    plt.title('Objective')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig('Objective' + str(i+1))


