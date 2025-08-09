import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from tqdm import tqdm

def train(model, num_epochs, train_dl, valid_dl):
    print('Start training ...')
    loss_hist_train = []
    loss_hist_valid = []
    accuracy_hist_train = []
    accuracy_hist_valid = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for x_batch, y_batch in tqdm(train_dl, desc='Training'):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            train_acc += is_correct.sum().item()
        
        loss_hist_train.append(train_loss / len(train_dl.dataset))
        accuracy_hist_train.append(train_acc / len(train_dl.dataset))

        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(valid_dl, desc='Validation'):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())

                valid_loss += loss.item() * y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float()
                valid_acc += is_correct.sum().item()
        
        loss_hist_valid.append(valid_loss / len(valid_dl.dataset))
        accuracy_hist_valid.append(valid_acc / len(valid_dl.dataset))
        
        print(f'\nEpoch {epoch+1}/{num_epochs}\n'
              f'Train Accuracy: {accuracy_hist_train[epoch]:.4f}, Validation Accuracy: '
              f'{accuracy_hist_valid[epoch]:.4f}')
        print(f'Train Loss: {loss_hist_train[epoch]:.4f}, '
              f'Validation Loss: {loss_hist_valid[epoch]:.4f}')
        
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved. (Epoch: {epoch+1})')
        
    return loss_hist_train, loss_hist_valid, \
           accuracy_hist_train, accuracy_hist_valid


get_smile = lambda attr: attr[31]

transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])
transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

image_path = '../chapter12/'
celeba_train_dataset = torchvision.datasets.CelebA(
    image_path, split='train',
    target_type='attr', download=False,
    transform=transform_train, target_transform=get_smile
)
celeba_valid_dataset = torchvision.datasets.CelebA(
    image_path, split='valid',
    target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)
celeba_test_dataset = torchvision.datasets.CelebA(
    image_path, split='test',
    target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)

celeba_train_dataset = Subset(celeba_train_dataset,
                              torch.arange(16000))
celeba_valid_dataset = Subset(celeba_valid_dataset,
                              torch.arange(1000))

batch_size = 32
torch.manual_seed(1)

train_dl = DataLoader(celeba_train_dataset,
                      batch_size, shuffle=True)
valid_dl = DataLoader(celeba_valid_dataset,
                      batch_size, shuffle=False)
test_dl = DataLoader(celeba_test_dataset,
                     batch_size, shuffle=False)

print(f'len(celeba_train_dataset) = {len(celeba_train_dataset)}')
print(f'type(celeba_train_dataset) = {type(celeba_train_dataset)}')
print(f'type(celeba_train_dataset[0]) = {type(celeba_train_dataset[0])}')
print(f'type(celeba_train_dataset[0][0]) = {type(celeba_train_dataset[0][0])}')
print(f'type(celeba_train_dataset[0][1]) = {type(celeba_train_dataset[0][0])}')

print()
print(f'celeba_train_dataset[0][0].shape = {celeba_train_dataset[0][0].shape}')
print(f'celeba_train_dataset[0][1].shape = {celeba_train_dataset[0][1].shape}')

print()
print(f'len(celeba_train_dataset[0][0]) = {len(celeba_train_dataset[0][0])}')
print(f'celeba_train_dataset[0][1].dim() = {celeba_train_dataset[0][1].dim()}')
print(f'len(celeba_train_dataset) / batch_size = {len(celeba_train_dataset)} / {batch_size} = {len(celeba_train_dataset) / batch_size}')
print(f'len(train_dl) = {len(train_dl)}')

model = nn.Sequential()
model.add_module(
    'conv1',
    nn.Conv2d(
        in_channels=3, out_channels=32,
        kernel_size=3, padding=1
    )
)
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout1', nn.Dropout(p=0.5))

model.add_module(
    'conv2',
    nn.Conv2d(
        in_channels=32, out_channels=64,
        kernel_size=3, padding=1
    )
)
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout2', nn.Dropout(p=0.5))

model.add_module(
    'conv3',
    nn.Conv2d(
        in_channels=64, out_channels=128,
        kernel_size=3, padding=1
    )
)
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))

model.add_module(
    'conv4',
    nn.Conv2d(
        in_channels=128, out_channels=256,
        kernel_size=3, padding=1
    )
)
model.add_module('relu4', nn.ReLU())
model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
model.add_module('flatten', nn.Flatten())

model.add_module('fc', nn.Linear(256, 1))
model.add_module('sigmoid', nn.Sigmoid())

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

model.to(device)
model_save_path = 'saved_model.pth'

torch.manual_seed(1)
num_epochs = 30
hist = train(model, num_epochs, train_dl, valid_dl)

accuracy_test = 0
model.eval()
with torch.no_grad():
    for x_batch, y_batch in tqdm(test_dl, desc='\n### Test'):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        pred = model(x_batch)[:, 0]
        is_correct = ((pred >= 0.5).float() == y_batch).float()
        accuracy_test += is_correct.sum()

accuracy_test /= len(test_dl.dataset)
print(f'Test Accuracy: {accuracy_test:.4f}\n')

x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train Loss')
ax.plot(x_arr, hist[1], '--<', label='Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()

pred = model(x_batch)[:, 0].cpu().detach().numpy() * 100
fig = plt.figure(figsize=(15, 7))
for j in range(10, 20):
    ax = fig.add_subplot(2, 5, j-10+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(x_batch[j].cpu().permute(1, 2, 0))
    if y_batch[j] == 1:
        label = 'Smile'
    else:
        label = 'Not Smile'
    ax.text(
        0.5, -0.15,
        f'GT: {label:s}\nP(Smile)={pred[j]:.0f}%',
        size=16,
        horizontalalignment = 'center',
        verticalalignment = 'center',
        transform=ax.transAxes
    )
plt.show()