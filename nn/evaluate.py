import numpy as np
import matplotlib.pyplot as plt
from model import forward_prop, load_data, load_model
from preprocess_data import load_and_transform_custom_data_to_npz


def test_model(modelname):
    w1, b1, w2, b2, w3, b3 = load_model(modelname)
    _, _, X_test, Y_test = load_data()
    _, _, _, _, _, A3 = forward_prop(X_test, w1, b1, w2, b2, w3, b3)
    index = np.random.randint(0, X_test.shape[0])
    print_prediction(index, X_test, Y_test, A3)
    acc, correct_idx, false_idx = calculate_accuracy(
        index, X_test, Y_test, w1, b1, w2, b2, w3, b3)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Wrong samples: {false_idx}")


def test_model_on_custom_data(modelname, data_name):
    w1, b1, w2, b2, w3, b3 = load_model(modelname)
    img = load_and_transform_custom_data_to_npz(data_name)
    X_custom = img.reshape(1, 784)

    # forward pass
    _, _, _, _, _, A3 = forward_prop(X_custom, w1, b1, w2, b2, w3, b3)

    # prediction
    prediction = np.argmax(A3)

    plt.imshow(img)
    plt.title("Predicted: " + str(prediction) +
              " True: " + data_name.split(".")[0])
    plt.axis("off")
    plt.show()

    return prediction


def calculate_accuracy(index, X, Y, w1, b1, w2, b2, w3, b3):
    # we do the forward propagation to get the output of the network
    _, _, _, _, _, A3 = forward_prop(X, w1, b1, w2, b2, w3, b3)
    # we get the predicted labels by taking the index of the maximum value in the output layer
    predictions = np.argmax(A3, axis=1)
    # we iterate through the predictions and compare them with the true labels to calculate the accuracy
    acc = np.mean(predictions == Y)
    correct_idx = np.where(predictions == Y)[0].tolist()
    false_idx = np.where(predictions != Y)[0].tolist()

    return acc, correct_idx, false_idx


def print_prediction(index, X, Y, prediction):
    image = X[index].reshape(28, 28)
    predictions = np.argmax(prediction, axis=1)
    plt.imshow(image)
    plt.title("Sample " + str(index) + "\nPredicted: " +
              str(predictions[index]) + " True: " + str(Y[index]))
    plt.axis("off")
    plt.show()
