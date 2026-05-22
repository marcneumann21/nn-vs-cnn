import numpy as np
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import torch


def load_and_transform_MNIST_data_to_np_array():
    # loading the dataset in cache (~/.keras/datasets/mnist.npz)
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

    # flatten the data from (sample_size, 28, 28) to (sample_size, 784)
    train_X_flattened = train_X.reshape(
        train_X.shape[0], train_X.shape[1] * train_X.shape[2])
    test_X_flattened = test_X.reshape(
        test_X.shape[0], test_X.shape[1] * test_X.shape[2])

    # convert the data into dataframes for putting together the features and labels
    train_df = pd.DataFrame(train_X_flattened)
    train_df["label"] = train_Y
    test_df = pd.DataFrame(test_X_flattened)
    test_df["label"] = test_Y

    # convert the dataframes into numpy arrays
    train_np_array = np.array(train_df)
    test_np_array = np.array(test_df)

    # shuffling the data
    np.random.shuffle(train_np_array)
    np.random.shuffle(test_np_array)

    # split the data again into features and labels
    X_train = train_np_array[:, :-1]
    Y_train = train_np_array[:, -1]
    Y_test = test_np_array[:, -1]
    X_test = test_np_array[:, :-1]

    return X_train, Y_train, X_test, Y_test


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


def load_and_transform_custom_data_to_npz(data_name):
    # load and resize image
    img = cv2.imread("./custom_data/"+data_name, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)

    # 7. Save
    name = re.sub(r"\.[^.]+$", "", data_name)
    np.savez(f"./custom_data/preprocessed_{name}.npz", image=img)

    return img


def load_and_transform_custom_data_to_torch_tensor(data_name):
    img = cv2.imread("./custom_data/" + data_name, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)

    img_tensor = torch.tensor(img,  dtype=torch.float32)
    # [,28,28] -> [1(sample_size), 1(batch_size), 28, 28]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    return img_tensor


def print_npz(data_name):
    data = np.load("./custom_data/" + data_name)
    image = data["image"].reshape(28, 28)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def print_torch(X_train):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i].cpu().numpy())
    plt.show()
