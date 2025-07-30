import numpy as np
from sklearn.model_selection import train_test_split

X = np.load('mnist_data_X.npy')
y = np.load('mnist_target_y.npy')

X = ((X / 255.) - 0.5) * 2

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000,
    random_state=123, stratify=y_temp
)