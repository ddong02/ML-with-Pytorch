from PIL import Image

try:
	img = Image.open('lenna.bmp')
	img.show()
except FileNotFoundError:
	print('File not found.')
except Exception as e:
	print(f'Error occured: {e}')

# 다양한 정보도 출력할 수 있다.
print(f'Image format: {img.format}')
print(f'Image size: {img.size}')
print(f'Image mode: {img.mode}')

import matplotlib.pyplot as plt

try:
	img = plt.imread('lenna.bmp')
	plt.imshow(img)
	plt.show()
except FileNotFoundError:
	print('File not found.')
except Exception as e:
	print(f'Error occured: {e}')

