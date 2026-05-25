from train import train_model
from evaluate import test_model, test_model_on_custom_data

if __name__ == "__main__":
    modelname = "model2"
    iterations = 500
    alpha = 0.01

    train_model(iterations, alpha, modelname)
    for i in range(1, 10):
        test_model(modelname)
    for i in range(1, 10):
        test_model_on_custom_data(modelname, f"{i}.png")
