import torch
import torch.nn as nn

logits = torch.tensor([0.8])
probas = torch.sigmoid(logits)
target =  torch.tensor([1.0])
bce_loss_fn = nn.BCELoss()
bce_logit_loss_fn = nn.BCEWithLogitsLoss()
print(f'BCD (probas): {bce_loss_fn(probas, target):.4f}')
print(f'BCE (logits): {bce_logit_loss_fn(logits, target):.4f}')

logits = torch.tensor([[1.5, 0.8, 2.1]])
probas = torch.softmax(logits, dim=1)
target = torch.tensor([2])
cce_loss_fn = nn.NLLLoss()
cce_logits_loss_fn = nn.CrossEntropyLoss()
print(f'CCE (logits): {cce_logits_loss_fn(logits, target):.4f}')
print(f'CCE (probas) {cce_loss_fn(torch.log(probas), target):.4f}')