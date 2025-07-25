import sys
import os
# 현재 파일 기준 상위 디렉토리 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.plot_decision_regions2 import plot_decision_regions
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.svm import SVC

for c in range(1, 500, 2):
    print(f'C = {c}')
    svm = SVC(kernel='linear', C=c, random_state=1)
    svm.fit(X_train_std, y_train)

    plot_decision_regions(X_combined_std,
                        y_combined,
                        classifier=svm,
                        test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('figures/03_11.png', dpi=300)
    plt.show()