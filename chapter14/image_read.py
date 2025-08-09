import torch
from torchvision.io import read_image
img = read_image('example-image.png')
print(f'Image size: {img.shape}')
print(f'Channel size: {img.shape[0]}')
print(f'Image data type: {img.dtype}')

print(img)