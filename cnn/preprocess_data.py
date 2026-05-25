import numpy as np
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import torch


def load_and_transform_MNIST_data_to_torch_tensor():
    # loading the dataset in cache (~/.keras/datasets/mnist.npz)
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

    X_train = torch.tensor(train_X, dtype=torch.float32) / 255.0
    X_test = torch.tensor(test_X, dtype=torch.float32) / 255.0

    Y_train = torch.tensor(train_Y, dtype=torch.long)
    Y_test = torch.tensor(test_Y, dtype=torch.long)

    # add chanel dimension [sample_size,28,28] -> [sample_size,1,28,28]
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)
    print(X_train.shape)
    print(X_test.shape)

    # datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    # loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader


def print_torch(X_train):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i].cpu().numpy())
    plt.show()


def preprocess_image(img):
    # resize image
    img = cv2.resize(img, (28, 28))

    # invert to get white digit on black background
    img = cv2.bitwise_not(img)

    # dialation for thickness
    kernel = np.ones((3, 3))
    img = cv2.dilate(img, kernel, iterations=1)

    # gaussia blur
    # we use a small kernel (3,3) so it doesn't wash out the strength
    img = cv2.GaussianBlur(img, (3, 3), 1.5)

    # final normalization
    img = img.astype("float32") / 255.0
    return img


def load_and_transform_custom_data_to_torch_tensor(data_name):
    img = cv2.imread("./../custom_data/" + data_name, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)

    img_tensor = torch.tensor(img,  dtype=torch.float32)
    # [,28,28] -> [1(sample_size), 1(batch_size), 28, 28]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    return img_tensor
