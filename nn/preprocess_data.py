import numpy as np
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


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


def print_npz(data_name):
    data = np.load("./../custom_data/" + data_name)
    image = data["image"].reshape(28, 28)
    plt.imshow(image)
    plt.axis("off")
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


def load_and_transform_custom_data_to_npz(data_name):
    # load and resize image
    img = cv2.imread("./../custom_data/"+data_name, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)

    # 7. Save
    name = re.sub(r"\.[^.]+$", "", data_name)
    np.savez(f"./../custom_data/preprocessed_{name}.npz", image=img)

    return img
