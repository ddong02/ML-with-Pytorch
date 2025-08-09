import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
from tqdm import tqdm

def train(model, num_epochs, train_dl, valid_dl):
    print(f'Start Training ...')
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in tqdm(train_dl, desc='Training'):
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
            ).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in tqdm(valid_dl, desc='Validation'):
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1}/{num_epochs} '
              f'Train Accuracy {accuracy_hist_train[epoch]:.4f}, '
              f'Validation Accuracy {accuracy_hist_valid[epoch]:.4f}')
    
    print(f'\nTraining finished ...')    
    print(f'Last metrics are ...')
    print(f'Train Loss: {loss_hist_train[epoch]}, Validation Loss: {loss_hist_valid[epoch]}')
    print(f'Train Accuracy: {accuracy_hist_train[epoch]}, Validation Accuracy: {accuracy_hist_valid[epoch]}')
    
    torch.save(model.state_dict(), model_save_path)
    print(f'\nModel saved in {model_save_path}')
    
    return loss_hist_train, loss_hist_valid, \
        accuracy_hist_train, accuracy_hist_valid

image_path = './'
model_save_path = 'model_state_dict.pth'
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=True
)
mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path, train=False,
    transform=transform, download=False
)
print(f'type(mnist_train_dataset[0]) = {type(mnist_train_dataset[0])}')
print(f'mnist_train_dataset[0][0].shape = {mnist_train_dataset[0][0].shape}')
print(f'mnist_train_dataset[0][1] = {mnist_train_dataset[0][1]}')
print(f'len(mnist_train_dataset) = {len(mnist_train_dataset)}')
batch_size = 64
print(f'len(mnist_train_dataset) / batch_size = {len(mnist_train_dataset) / batch_size}')

torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
print(f'len(train_dl) = {len(train_dl)}')
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
torch.manual_seed(1)
num_epochs = 20
hist = train(model, num_epochs, train_dl, valid_dl)

x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 4))

ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '-s', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], 's', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show()