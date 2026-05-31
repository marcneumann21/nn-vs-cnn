# Neural Network (NumPy) vs Convolutional Neural Network (PyTorch)

This project implements handwritten digit classification on the MNIST dataset using:

- A fully-connected Neural Network implemented from scratch using NumPy
- A Convolutional Neural Network implemented using PyTorch
- MNIST dataset

The project supports training, evaluation, saving/loading models, and testing on custom images.

---

# ⚠️ Requirements

This project **must use Python 3.11**.

Using newer Python versions (e.g. 3.12+) may cause compatibility issues with ML libraries such as PyTorch, NumPy, or TensorFlow.

---

# 🛠 Setup

## 1. Install Python 3.11

On macOS (Homebrew):

```bash
brew install python@3.11
```

On Linux (apt):

```bash
sudo apt install python@3.11
```
---

## 2. Create Virtual Environment

From the project root:

```bash
python3.11 -m venv .venv
```

---

## 3. Activate Virtual Environment

```bash
source .venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run the Project

You can run either implementation:

## ▶ NN

Run in the nn/ folder:

```bash
python3 main.py
```

## ▶ CNN

run in the cnn/ folder:

```bash
python3 main.py
```

---

# 🖼 Custom Data

You can test your own handwritten digit images.

Place images here:

custom_data/{number}.png

---

# ✨ Features

- MNIST digit classification (0–9)
- Fully-connected NN from scratch (NumPy)
- CNN using PyTorch
- Training loop with accuracy tracking
- Model saving & loading
- Custom image inference
- GPU support (MPS / CUDA if available)
