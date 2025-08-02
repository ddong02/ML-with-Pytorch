import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def net_input(X, w):
    return np.dot(X, w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

X = np.array([1, 1.4, 2.5])
w = np.array([0.4, 0.3, 0.5])

W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])

A = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas = logistic(Z)

# print('P(y=1|x) = %.3f' % logistic_activation(X, w))

# print('Final input: \n', Z)
# print('Output units: \n', y_probas)

y_probas = softmax(Z)

# print('Probablity:\n', y_probas)
# print(np.sum(y_probas))

t = torch.softmax(torch.from_numpy(Z), dim=0)
# print(t)

z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)
plt.ylim([-1.5, 1.5])
plt.xlabel('Net input $z$')
plt.ylabel('Activation $\sigma(z)$')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')

plt.plot(z, tanh_act,
         linewidth=3, linestyle='--',
         label='Tanh')
plt.plot(z, log_act,
         linewidth=3,
         label='Logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()