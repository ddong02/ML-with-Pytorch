### Single Layer Perceptron (SLP)

import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def predict(self, X):   ### 입력값(net.input)이 0 이상이면 1을 반환, 아니면 0을 반환
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def net_input(self, X): ### z = wx + b
        return np.dot(X, self.w_) + self.b_

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state) # SEED 고정 → 결과 재현 가능
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.w_ += update * x_i
                self.b_ += update
                errors += int(update != 0.0)

            self.errors_.append(errors)
            

        return self
