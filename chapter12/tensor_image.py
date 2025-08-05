from torchvision.io import read_image
import matplotlib.pyplot as plt

image_tensor = read_image('image.png')

print(f'image tensor shape: {image_tensor.shape}')
# C x H x W 형태의 텐서를 H x W x C 형태로 변환해줘야 함
converted_image = image_tensor.permute(1, 2, 0)

plt.imshow(converted_image)
plt.show()
