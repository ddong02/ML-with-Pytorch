from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

file_name = 'example-image.png'

input_image = Image.open(file_name).convert('RGB')

input_tensor = F.to_tensor(input_image)
output_tensor = F.vflip(input_tensor)
output_tensor2pil = F.to_pil_image(output_tensor)

output_pil = F.vflip(input_image)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(output_tensor2pil)
plt.title('output tensor to PIL')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output_pil)
plt.title('flipped PIL direct')
plt.axis('off')

plt.show()
