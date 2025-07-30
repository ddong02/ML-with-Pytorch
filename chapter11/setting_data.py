import numpy as np
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) # as_frame=False를 사용하여 NumPy 배열로 받습니다.

np.save('data/mnist_data_X.npy', X)
np.save('data/mnist_target_y.npy', y)