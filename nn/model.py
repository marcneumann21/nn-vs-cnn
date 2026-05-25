import numpy as np
import os
from preprocess_data import load_and_transform_MNIST_data_to_np_array


# Initialize the weights and biases
def init_params():
    # first layer 784 input neurons (input layer), 64 output neurons (1. hidden layer)
    w1 = np.random.randn(784, 64) * 0.01
    b1 = np.zeros((1, 64))

    # second layer: 64 input neurons (1. hidden layer), 32 output neurons (2. hidden layer)
    w2 = np.random.randn(64, 32) * 0.01
    b2 = np.zeros((1, 32))

    # third layer: 32 input neurons (2. hidden layer), 10 output neurons (output layer)
    w3 = np.random.randn(32, 10) * 0.01
    b3 = np.zeros((1, 10))
    return w1, b1, w2, b2, w3, b3

# Forward propagation with X as input and the weights and biases as parameters


def forward_prop(X, w1, b1, w2, b2, w3, b3):
    # First layer output
    Z1 = np.dot(X, w1) + b1
    A1 = relu(Z1)

    # Second layer output
    Z2 = np.dot(A1, w2) + b2
    A2 = relu(Z2)

    # Third layer output
    Z3 = np.dot(A2, w3) + b3
    # final output with softmax activation function
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Backward propagation with X as input, the weights and biases as parameters and the output of the forward propagation


def backward_prop(X, Z1, A1, Z2, A2, Z3, A3, Y, w2, w3):
    # Number of samples
    m = Y.shape[0]

    # Convert the labels to one-hot encoding
    one_hot_Y = one_hot(Y)

    # Calculate the gradients for the output layer
    dZ3 = A3 - one_hot_Y
    # The gradients for the weights and biases of the this layer are calculated with the output of the layer before
    dw3 = 1 / m * A2.T.dot(dZ3)
    # The gradients for the biases are calculated with the sum of the gradients of the output layer
    # We do axis=0 because we want to iterate column-wise(the features)
    db3 = 1 / m * np.sum(dZ3, axis=0, keepdims=True)

    # Calculate the gradients for the second hidden layer
    dA2 = np.dot(dZ3, w3.T)
    dZ2 = dA2 * deriv_relu(Z2)
    dw2 = 1 / m * A1.T.dot(dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

    # Calculate the gradients for the first hidden layer
    dA1 = np.dot(dZ2, w2.T)
    dZ1 = dA1 * deriv_relu(Z1)
    # Here we use the input X because the output of the input layer is the input data itself
    dw1 = 1 / m * X.T.dot(dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)
    return dw1, db1, dw2, db2, dw3, db3

# Update the weights and biases with the gradients and the learning rate alpha


def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3
    return w1, b1, w2, b2, w3, b3

# Function to convert the labels to one-hot encoding


def one_hot(Y):
    # First create a zero matrix of shape (number of samples, number of classes)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))

    # Fill the matrix with 1s at the appropriate positions
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y


def relu(Z):
    # return the max of 0 and Z
    return np.maximum(0, Z)


def deriv_relu(Z):
    # the derivative of relu is 1 for Z > 0 and 0 for Z <= 0
    return (Z > 0)


def softmax(Z):
    # to prevent overflow we subtract the max of Z from Z before applying the exponential function
    # we do axis=1 because we want to iterate row-wise (the samples)
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# the gradient descent function that combines the forward propagation,
# backward propagation and parameter update


def gradient_descent(X, Y, w1, b1, w2, b2, w3, b3, alpha):
    # First we do the forward propagation to get the output of the network
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(X, w1, b1, w2, b2, w3, b3)

    # Then we do the backward propagation to get the gradients of the weights and biases
    dw1, db1, dw2, db2, dw3, db3 = backward_prop(
        X, Z1, A1, Z2, A2, Z3, A3, Y, w2, w3)

    # Finally we update the weights and biases with the gradients and the learning rate alpha
    w1, b1, w2, b2, w3, b3 = update_params(
        w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
    return w1, b1, w2, b2, w3, b3


def load_data():
    # Load and transform the MNIST data to NumPy arrays
    X_train, Y_train, X_test, Y_test = load_and_transform_MNIST_data_to_np_array()
    return X_train, Y_train, X_test, Y_test


def load_model(modelname):
    if os.path.exists("./../models/nn/" + modelname + ".npz"):
        print(f"Loading model {modelname}...")
        data = np.load("./../models/nn/" + modelname + ".npz")
        w1 = data["w1"]
        b1 = data["b1"]
        w2 = data["w2"]
        b2 = data["b2"]
        w3 = data["w3"]
        b3 = data["b3"]
        return w1, b1, w2, b2, w3, b3
    else:
        w1, b1, w2, b2, w3, b3 = init_params()
        print(f"Model {modelname} not found.")
        print(f"Creating new model {modelname}...")
        np.savez("./../models/nn/" + modelname, w1=w1,
                 b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
        return w1, b1, w2, b2, w3, b3
