import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)

class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

joint_dataset = JointDataset(t_x, t_y)

from torch.utils.data import TensorDataset
joint_dataset = TensorDataset(t_x, t_y)

torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)

# for i, batch in enumerate(data_loader, 1):
#     print(f'Batch {i}:', 'x:', batch[0],
#           '\n   y:', batch[1])

# for epoch in range(2):
#     print(f'Epoch {epoch+1}')
#     for i, batch in enumerate(data_loader, 1):
#         print(f'Batch {i}:', 'x:', batch[0],
#         '\n\t   y:', batch[1])
        
import pathlib
imgdir_path = pathlib.Path('cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img = Image.open(file)
    print(f'Image shape:{np.array(img).shape}')
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

plt.tight_layout()
# plt.show()

labels = [1 if 'dog' in
          os.path.basename(file) else 0
          for file in file_list]
# print(labels)

class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img =  Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.labels)

image_dataset = ImageDataset(file_list, labels)
# for file, label in image_dataset:
#     print(file, label)
    
import torchvision.transforms as transforms
img_height, img_width = 80, 120
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height, img_width))
])

image_dataset = ImageDataset(file_list, labels, transform)

fig = plt.figure(figsize=(10, 6))
for i, example in enumerate(image_dataset):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0].numpy().transpose((1, 2, 0)))
    print(f'Image shape:{example[0].shape}')
    ax.set_title(f'{example[1]}', size=15)

plt.tight_layout()
# plt.show()