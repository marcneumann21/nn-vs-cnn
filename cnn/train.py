import torch
import os
from torch import nn, optim
from preprocess_data import load_and_transform_MNIST_data_to_torch_tensor
from model import CNN, load_model


def train_model(iterations, alpha, modelname):
    model, device = load_model(modelname)

    train_dataloader, _ = load_and_transform_MNIST_data_to_torch_tensor()

    criterion = nn.CrossEntropyLoss()
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

    # Adam optimizer to update the weights of the model
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    print(f"Training new model {modelname}...")
    for epoch in range(iterations):
        print(f"\nEpoch [{epoch + 1}/{iterations}]")

        model.train()

        correct = 0
        total = 0
        running_loss = 0

        for data, targets in train_dataloader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy tracking
            _, preds = torch.max(scores, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item()

        acc = correct / total
        avg_loss = running_loss / len(train_dataloader)

        print(f"Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), "./../models/cnn/"+modelname)


def plot_feature_maps(act, title):
    num_maps = min(16, act.shape[0])

    plt.figure(figsize=(6, 6))
    for i in range(num_maps):
        plt.subplot(4, 4, i + 1)
        plt.imshow(act[i], cmap="viridis")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
