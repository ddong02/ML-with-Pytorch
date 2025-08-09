import torch
import torch.nn as nn
loss_fn = nn.BCELoss()
loss = loss_fn(torch.tensor([0.9]), torch.tensor([1.0]))
l2_lambda = 0.001
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=5,
                       kernel_size=5)
l2_penalty = l2_lambda * sum(
    [(p**2).sum() for p in conv_layer.parameters()]
)
loss_with_penalty = loss + l2_penalty
linear_layer = nn.Linear(10, 16)
l2_penalty = l2_lambda * sum(
    [(p**2).sum() for p in linear_layer.parameters()]
)
loss_with_penalty = loss + l2_penalty