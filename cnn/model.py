import os
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1st conv layer
        # 'same' is padding=1 for k=3
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1)
        # 2nd conv layer
        # 'same' is padding=2 for k=5
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1)

        # max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))

        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor
        x = x.reshape(x.shape[0], -1)

        # FC1
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)

        return x


def load_data():
    # Load and transform the MNIST data to PyTorch tensors
    train_dataloader, test_dataloader = load_and_transform_MNIST_data_to_torch_tensor()
    return train_dataloader, test_dataloader


def load_model(modelname):
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists("./../models/cnn/"+modelname):
        print(f"Loading model {modelname}...")
        model.load_state_dict(
            torch.load("./../models/cnn/" + modelname,
                       map_location=device,
                       weights_only=True
                       )
        )
    else:
        print(f"Creating new model {modelname}...")
        model = CNN().to(device)

    return model, device
