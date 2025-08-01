import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from NeuralNetMLP import NeuralNetMLP
from NeuralNetMLP import int_to_onehot

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

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
     mse, correct_pred, num_examples = 0.0, 0, 0
     minibatch_gen = minibatch_generator(X, y, minibatch_size)
     
     for i, (features, targets) in enumerate(minibatch_gen):
          _, probas = nnet.forward(features)
          predicted_labels = np.argmax(probas, axis=1)
          onehot_targets = int_to_onehot(
               targets, num_labels=num_labels
          )
          loss = np.mean((onehot_targets - probas) ** 2)
          correct_pred += (predicted_labels == targets).sum()
          num_examples += targets.shape[0]
          mse += loss
     
     mse = mse/i
     acc = correct_pred / num_examples
     
     return mse, acc

def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1, minibatch_size=100):
     epoch_loss = []
     epoch_train_acc = []
     epoch_valid_acc = []
     
     for e in range(num_epochs):
          minibatch_gen = minibatch_generator(
               X_train, y_train, minibatch_size
          )
          for X_train_mini, y_train_mini in minibatch_gen:
               a_h, a_out = model.forward(X_train_mini)

               d_loss__d_w_out, d_loss__d_b_out, \
               d_loss__d_w_h, d_loss__d_b_h = \
                    model.backward(X_train_mini, a_h, a_out,
                                   y_train_mini)
               
               model.weight_h -= learning_rate * d_loss__d_w_h
               model.bias_h -= learning_rate * d_loss__d_b_h
               model.weight_out -= learning_rate * d_loss__d_w_out
               model.bias_out -= learning_rate * d_loss__d_b_out
          
          train_mse, train_acc = compute_mse_and_acc(
               model, X_train, y_train
          )
          valid_mse, valid_acc = compute_mse_and_acc(
               model, X_valid, y_valid
          )
          train_acc, valid_acc = train_acc*100, valid_acc*100
          epoch_train_acc.append(train_acc)
          epoch_valid_acc.append(valid_acc)
          epoch_loss.append(train_mse)

          print(f'Epoch: {e+1:03d}/{num_epochs:03d}'
                f'| Train MSE: {train_mse:.2f} '
                f'| Train Acc: {train_acc:.2f}% '
                f'| Valid Acc: {valid_acc:.2f}%')
          
     return epoch_loss, epoch_train_acc, epoch_valid_acc


if __name__ == '__main__':
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

     np.random.seed(123)
     epochs = 50
     learning_rate = 0.1
     epoch_loss, epoch_train_acc, epoch_valid_acc = train(
          model, X_train, y_train, X_valid, y_valid, epochs, learning_rate
     )
     
     loss_acc_df = pd.DataFrame({
          'epoch_loss': epoch_loss,
          'epoch_train_acc': epoch_train_acc,
          'epoch_valid_acc': epoch_valid_acc
     })
     
     file_path = './train_result/loss_acc_result_no_index.csv'
     loss_acc_df.to_csv(file_path, index=False)
     
     test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
     print(f'Test Accuracy: {test_acc*100: .2f}%')
     
     X_test_subset = X_test[:1000, :]
     y_test_subset = y_test[:1000]
     _, probas = model.forward(X_test_subset)
     test_pred = np.argmax(probas, axis=1)
     
     misclassified_image = \
          X_test_subset[y_test_subset != test_pred][:25]
     misclassified_labels = test_pred[y_test_subset != test_pred][:25]
     correct_labels = y_test_subset[y_test_subset != test_pred][:25]
     
     fig, ax = plt.subplots(nrows=5, ncols=5,
                            sharex=True, sharey=True,
                            figsize=(8, 8))
     ax = ax.flatten()
     for i in range(25):
          img = misclassified_image[i].reshape(28, 28)
          ax[i].imshow(img, cmap='Greys', interpolation='nearest')
          ax[i].set_title(f'{i+1})' 
                          f'True: {correct_labels[i]}\n'
                          f'Predicted: {misclassified_labels[i]}')
     ax[0].set_xticks([])
     ax[0].set_yticks([])
     plt.tight_layout()
     plt.show()