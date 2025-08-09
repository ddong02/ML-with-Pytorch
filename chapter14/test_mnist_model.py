import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
from tqdm import tqdm

image_path = './'
model_save_path = 'model_state_dict.pth'
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path, train=False,
    transform=transform, download=False
)

model = nn.Sequential()
model.add_module(
    'conv1',
    nn.Conv2d(
        in_channels=1, out_channels=32,
        kernel_size=5, padding=2
    )
)
model.add_module(
    'ReLU1',
    nn.ReLU()
)
model.add_module(
    'pool1',
    nn.MaxPool2d(kernel_size=2)
)
model.add_module(
    'conv2',
    nn.Conv2d(in_channels=32, out_channels=64,
              kernel_size=5, padding=2)
)
model.add_module(
    'ReLU2',
    nn.ReLU()
)
model.add_module(
    'pool2',
    nn.MaxPool2d(kernel_size=2)
)
model.add_module(
    'flatten',
    nn.Flatten()
)
model.add_module(
    'fc1',
    nn.Linear(3136, 1024)
)
model.add_module(
    'ReLU3',
    nn.ReLU()
)
model.add_module(
    'dropout',
    nn.Dropout(p=0.5)
)
model.add_module(
    'fc2',
    nn.Linear(1024, 10)
)

state_dict = torch.load(model_save_path)
model.load_state_dict(state_dict)
model.eval()

pred = model(mnist_test_dataset.data.unsqueeze(1) / 255.)

is_correct = (
    torch.argmax(pred, dim=1) == mnist_test_dataset.targets
).float()

misclassified_indices = torch.where(
    torch.argmax(pred, dim=1) != mnist_test_dataset.targets
)[0]

print(f'Test Accuracy: {is_correct.mean():.4f}')

fig = plt.figure(figsize=(12,8))

for i in range(25):
    ax = fig.add_subplot(5, 5, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    idx = misclassified_indices[i].item()
    img = mnist_test_dataset[idx][0][0, :, :]
    true_label = mnist_test_dataset.targets[idx].item()
    predicted_label = torch.argmax(pred[idx]).item()
    ax.imshow(img, cmap='gray_r')
    ax.set_title(f'True: {true_label}\nPred: {predicted_label}', color='red')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    img = mnist_test_dataset[i][0][0, :, :]
    pred = model(img.unsqueeze(0).unsqueeze(1))
    y_pred = torch.argmax(pred)
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, y_pred.item(),
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()