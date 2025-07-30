import numpy as np
from sklearn.model_selection import train_test_split

from NeuralNetMLP import NeuralNetMLP
from NeuralNetMLP import int_to_onehot

num_epochs = 50
minibatch_size = 100

def minibatch_generator(X, y, minibatch_size):
     indices = np.arange(X.shape[0])
     np.random.shuffle(indices)
     for start_idx in range(0, indices.shape[0] - minibatch_size + 1,
                            minibatch_size):
          batch_idx = indices[start_idx:start_idx + minibatch_size]
          yield X[batch_idx], y[batch_idx]

def mse_loss(targets, probas, num_labels=10):
     onehot_targets = int_to_onehot(
          targets, num_labels=num_labels
     )
     return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
     return np.mean(predicted_labels == targets)

model = NeuralNetMLP(num_features=28*28,
                     num_hidden=50,
                     num_classes=10)

X = np.load('data/mnist_data_X.npy')
y = np.load('data/mnist_target_y.npy')

X = ((X / 255.) - 0.5) * 2

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000,
    random_state=123, stratify=y_temp
)

_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'validation set MSE: {mse:.1f}')

predicted_labels = np.argmax(probas, axis=1)
### ↑ 행을 기준으로 최대값 추출 shape = (5000, )
acc = accuracy(y_valid, predicted_labels)
print(f'validation set Accuracy: {acc*100:.1f}%')