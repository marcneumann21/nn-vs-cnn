import os
import numpy as np
from model import init_params, gradient_descent, load_data, load_model
from evaluate import calculate_accuracy


def train_model(iterations, alpha, modelname):
    w1, b1, w2, b2, w3, b3 = load_model(modelname)

    X_train, Y_train, _, _ = load_data()

    # training loop
    for i in range(1, iterations+1):
        w1, b1, w2, b2, w3, b3 = gradient_descent(
            X_train, Y_train, w1, b1, w2, b2, w3, b3, alpha)

        if i % 50 == 0:
            print(f"Epoch {i}/{iterations} completed")
            acc, _, _ = calculate_accuracy(
                i, X_train, Y_train, w1, b1, w2, b2, w3, b3)
            np.savez("./../models/nn/" + modelname, w1=w1,
                     b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
            print(f"Training Accuracy: {acc*100:.2f}%")

    return w1, b1, w2, b2, w3, b3
