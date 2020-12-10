""" Usage:
    X is a numpy array of shape (num_features, num_samples), i.e. each COLUMN is a training sample
    Y is a numpy array of shape (label_range=10, num_samples), i.e. each COLUMN is a one hot encoded vector
    layer_sizes = [num_features, 100, 50, label_range]
    layer_activation_functions = [activations.sigmoid] * len(layer_sizes)
    layer_derivative_functions = [activations.sigmoid_gradient] * len(layer_sizes)
    nn = NeuralNetwork(layer_sizes, layer_activation_functions, layer_derivative_functions)
    nn.train(X, Y, learning_rate, batch_size, num_epochs)
    accuracy = nn.accuracy(X, Y)
    """

import numpy as np
# import activations
import h5py as h5
from sklearn.model_selection import train_test_split


# cross-entropy
def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))


def sigmoid_gradient(X):
    return sigmoid(X) * (1.0 - sigmoid(X))


def loss_function(Y, A):
    return Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A)


def loss_derivative_function(Y, A):
    return ((1.0 - Y) / (1.0 - A) - (Y / A))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax2(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def softmax_gradient(s):
    # input s is softmax value of the original input x. Its shape is (1,n)
    # e.i. s = np.array([0.3,0.7]), x = np.array([0,1])

    # make the matrix whose size is n^2.
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1 - s[i])
            else:
                jacobian_m[i][j] = -s[i] * s[j]
    return jacobian_m


class NeuralNetwork(object):

    def __init__(self, layer_sizes, layer_activation_functions, layer_derivative_functions):
        self.layer_sizes = layer_sizes
        self.layer_activation_functions = layer_activation_functions
        self.layer_derivative_functions = layer_derivative_functions
        self.initialize_parameters()

    def initialize_parameters(self):
        num_layers = len(self.layer_sizes)
        W = {}
        Z = {}
        A = {}
        B = {}
        dA = {}
        dZ = {}
        dW = {}
        dB = {}
        for i in range(1, num_layers):
            W[i] = np.random.uniform(-1.0, 1.0, (self.layer_sizes[i], self.layer_sizes[i - 1]))
            B[i] = np.zeros((self.layer_sizes[i], 1))
            dW[i] = np.zeros(W[i].shape)
            dB[i] = np.zeros(B[i].shape)
        self.W = W
        self.Z = Z
        self.A = A
        self.B = B
        self.dA = dA
        self.dZ = dZ
        self.dW = dW
        self.dB = dB

    def feedforward(self):
        num_layers = len(self.layer_sizes)
        for i in range(1, num_layers):
            self.Z[i] = np.dot(self.W[i], self.A[i - 1], ) + self.B[i]
            self.A[i] = self.layer_activation_functions[i](self.Z[i])

    def backprop(self):
        num_layers = len(self.layer_sizes)
        num_examples = self.A[0].shape[1]
        for i in reversed(range(1, num_layers)):
            self.dZ[i] = self.dA[i] * self.layer_derivative_functions[i](self.Z[i])
            self.dW[i] = np.dot(self.dZ[i], self.A[i - 1].T) / num_examples
            self.dB[i] = np.sum(self.dZ[i], axis=1, keepdims=True) / num_examples
            self.dA[i - 1] = np.dot(self.W[i].T, self.dZ[i])

    def update_parameters(self, learning_rate):
        num_layers = len(self.layer_sizes)
        for i in range(1, num_layers):
            self.W[i] = self.W[i] - learning_rate * self.dW[i]
            self.B[i] = self.B[i] - learning_rate * self.dB[i]

    def single_train_step(self, X, Y, learning_rate):
        self.A[0] = X
        self.feedforward()
        num_layers = len(self.layer_sizes)
        loss = loss_function(Y, self.A[num_layers - 1])
        loss_derivative = loss_derivative_function(Y, self.A[num_layers - 1])
        self.dA[num_layers - 1] = loss_derivative
        self.backprop()
        self.update_parameters(learning_rate)

    def train(self, X, Y, learning_rate, batch_size, num_epochs):
        # np.random.shuffle(X)
        num_examples = X.shape[0]
        num_features = X.shape[1]
        num_batches = num_examples / batch_size
        for epoch in range(num_epochs):
            accuracy = self.accuracy(X, Y)
            print(accuracy)
            for batch in range(batch_size):
                starting_index = batch * batch_size
                stopping_index = starting_index + batch_size
                mini_batch_X = X[:, starting_index:stopping_index]
                mini_batch_Y = Y[:, starting_index:stopping_index]
                self.single_train_step(mini_batch_X, mini_batch_Y, learning_rate)

    def predict(self, X):
        num_layers = len(self.layer_sizes)
        self.A[0] = X
        self.feedforward()
        probabilities = self.A[num_layers - 1]
        predicted_labels = probabilities.argmax(axis=0)
        return predicted_labels

    def accuracy(self, X, Y):
        predicted_labels = self.predict(X)
        ground_truth = Y.argmax(axis=0)
        is_correct = (predicted_labels == ground_truth)
        num_correct_predictions = sum(is_correct)
        num_examples = X.shape[1]
        accuracy = num_correct_predictions / float(num_examples)
        return accuracy


def load_h5py(filename):
    with h5.File(filename, 'r') as hf:
        x = hf['X'][:]
        y = hf['Y'][:]
    x = np.array([i.flatten() for i in x])
    # y = y.reshape(len(y),1)
    y = (y == 7).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    return x_train, y_train, x_test, y_test


def program(batch_size, learning_rate, num_epochs):
    x_train, y_train, x_test, y_test = load_h5py('MNIST_Subset.h5')

    # Converting y_train to hot encode
    b = np.zeros((len(y_train), 2))
    b[np.arange(len(y_train)), y_train] = 1
    y_train = b
    y_train = np.transpose(y_train)

    # Converting y_train to hot encode
    b = np.zeros((len(y_test), 2))
    b[np.arange(len(y_test)), y_test] = 1
    y_test = b
    y_test = np.transpose(y_test)
    x_train, x_test = np.transpose(x_train), np.transpose(x_test)

    layer_sizes = [784, 100, 50, 50, 2]
    layer_activation_functions = [sigmoid] * len(layer_sizes)
    layer_derivative_functions = [sigmoid_gradient] * len(layer_sizes)
    nn = NeuralNetwork(layer_sizes, layer_activation_functions, layer_derivative_functions)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    nn.train(x_train, y_train, learning_rate, batch_size, num_epochs)
    accuracy = nn.accuracy(x_test, y_test)
    return accuracy


def k_fold(k, batch_size, learning_rate, num_epochs):
    accuracy_mean = 0
    for i in range(k):
        accuracy_mean = float(accuracy_mean) + program(batch_size, learning_rate, num_epochs) / float(k)
    print("The mean accuracy is :" + str(accuracy_mean))
    return accuracy_mean


def grid_search():
    batch_size_list = [10, 20, 30, 40]
    learning_rate_list = [0.01, 0.05, 0.1, 1.5]
    num_epochs_list = [10, 20, 30, 40]
    batch_size = 20
    learning_rate = 0.1
    num_epochs = 10
    highest_accuracy = 0
    for i in batch_size_list:
        print("Batch is " + str(i))
        cur_acc = k_fold(i, 0.75, 30)
        if cur_acc > highest_accuracy:
            batch_size = i
            highest_accuracy = cur_acc
    highest_accuracy = 0
    for i in num_epochs_list:
        print("num_epochs is " + str(i))
        cur_acc = k_fold(batch_size, 0.1, i)
        if cur_acc > highest_accuracy:
            num_epochs = i
            highest_accuracy = cur_acc
    highest_accuracy = 0
    for i in learning_rate_list:
        print("learning_rate is " + str(i))
        cur_acc = k_fold(batch_size, i, num_epochs)
        if cur_acc > highest_accuracy:
            learning_rate = i
            highest_accuracy = cur_acc
    print(batch_size)
    print(learning_rate)
    print(num_epochs)


def grid_search2():
    batch_size_list = [10, 20, 30, 40]
    learning_rate_list = [0.01, 0.05, 0.1, 1.5]
    num_epochs_list = [10, 20, 30, 40]
    max_acc = 0
    batch_size, learning_rate, num_epochs = 0, 0, 0
    for i in batch_size_list:
        for j in learning_rate_list:
            for k in num_epochs_list:
                cur_acc = k_fold(4, i, j, k)
                if cur_acc > max_acc:
                    max_acc = cur_acc
                    batch_size = i
                    learning_rate = j
                    num_epochs = k


program(40, 0.1, 200)



