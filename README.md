# MNIST Neural Network (NumPy)

This project implements a simple fully-connected neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset.

## Features
- Load MNIST dataset via tensorflow
- Manual implementation of forward and backward propagation
- Fully connected neural network with 2 hidden layers
- Model training using gradient descent
- Save and load model weights (`.npz`)
- Test single predictions and compute accuracy

## Model Architecture
- Input: 784 (28×28 images flattened)
- Hidden layer 1: 64 neurons
- Hidden layer 2: 32 neurons
- Output layer: 10 neurons (digits 0–9)
