import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU()
)

nn.init.xavier_uniform_(model[0].weight)
l1_weight = 0.01
l1_panalty = l1_weight * model[2].weight.abs().sum()

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)