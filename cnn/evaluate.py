import torch
import matplotlib.pyplot as plt
from model import CNN, load_model
from preprocess_data import load_and_transform_MNIST_data_to_torch_tensor, load_and_transform_custom_data_to_torch_tensor


def test_model(modelname):
    model, device = load_model(modelname)
    model.eval()

    _, test_dataloader = load_and_transform_MNIST_data_to_torch_tensor()

    correct = 0
    total = 0

    # calculate accuracy
    with torch.no_grad():
        for images, labels in test_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total

        # print random image
        images, labels = next(iter(test_dataloader))

        images = images.to(device)
        labels = labels.to(device)

        idx = torch.randint(0, images.size(0), (1,)).item()
        img = images[idx].unsqueeze(0)   # keep shape [1,1,28,28]
        label = labels[idx]

        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        img_show = img.cpu().squeeze()
        plt.imshow(img_show)
        plt.title("Predicted: " + str(preds.item()) +
                  " | True: " + str(labels[idx].item()))
        plt.axis("off")
        plt.show()

        print("Probabilities:", probs.cpu().numpy().round(2).tolist())
        print(f"Accuracy: {acc:.4f}")


def test_model_custom_data(modelname, image):
    model, device = load_model(modelname)
    model.eval()

    activations = {}
    # ---- hook function ----

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(get_activation("conv1"))
    model.conv1.register_forward_hook(get_activation("conv2"))

    images = load_and_transform_custom_data_to_torch_tensor(image)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        print("Predictions:", preds.item())
        print("Probabilities:", probs.cpu().numpy().round(2).tolist())

        img_show = images.cpu().squeeze()
        plt.imshow(img_show)
        plt.title("Predicted: " + str(preds.item()) +
                  " | True: " + image.split(".")[0])
        plt.axis("off")
        plt.show()

     # ---------------- CONV1 ----------------
    act1 = activations["conv1"][0].cpu()
    plot_feature_maps(act1, "Conv1 Activations")

    # ---------------- CONV2 ----------------
    act2 = activations["conv2"][0].cpu()
    plot_feature_maps(act2, "Conv2 Activations")


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
