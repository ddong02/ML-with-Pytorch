import torch
from torchvision.transforms import functional as F
from PIL import Image
import os
import matplotlib.pyplot as plt

file_name = 'example-image.png'

input_image = Image.open(file_name).convert('RGB')
print(f'{file_name} <- file loaded successfully.')

input_tensor = F.to_tensor(input_image)
print(f'image tensor shape: {input_tensor.shape}')

output_size = (300, 300)
output_tensor = F.center_crop(input_tensor, output_size)
print(f'output image tensor shape: {output_tensor.shape}')

output_image_pil = F.to_pil_image(output_tensor)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image_pil)
plt.title('Transformed Image')
plt.axis('off')

plt.show()