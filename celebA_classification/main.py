import torch
import torch.nn as nn
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from tqdm import tqdm

def get_dataloaders(image_path='./',
					batch_size=32,
					train_transform=None,
					eval_transform=None,
					target_transform=None):
	
	celeba_train_dataset = torchvision.datasets.CelebA(
		image_path, split='train',
		target_type='attr', download=False,
		transform=train_transform, target_transform=target_transform
	)
	celeba_valid_dataset = torchvision.datasets.CelebA(
		image_path, split='valid',
		target_type='attr', download=False,
		transform=eval_transform, target_transform=target_transform
	)
	celeba_test_dataset = torchvision.datasets.CelebA(
		image_path, split='test',
		target_type='attr', download=False,
		transform=eval_transform, target_transform=target_transform
	)

	celeba_train_dataset = Subset(celeba_train_dataset, torch.arange(16000))
	celeba_valid_dataset = Subset(celeba_valid_dataset, torch.arange(2000))

	train_dl = DataLoader(celeba_train_dataset, batch_size, shuffle=True)
	valid_dl = DataLoader(celeba_valid_dataset, batch_size, shuffle=False)
	test_dl = DataLoader(celeba_test_dataset, batch_size, shuffle=False)

	return train_dl, valid_dl, test_dl


def get_model():
	model = nn.Sequential()
	model.add_module('conv1', nn.Conv2d(
		in_channels=3, out_channels=32, kernel_size=3, padding=1
	))
	model.add_module('relu1', nn.ReLU())
	model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
	model.add_module('dropout1', nn.Dropout(p=0.5))
	model.add_module('conv2', nn.Conv2d(
		in_channels=32, out_channels=64, kernel_size=3, padding=1
	))
	model.add_module('relu2', nn.ReLU())
	model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
	model.add_module('dropout2', nn.Dropout(p=0.5))
	model.add_module('conv3', nn.Conv2d(
		in_channels=64, out_channels=128, kernel_size=3, padding=1
	))
	model.add_module('relu3', nn.ReLU())
	model.add_module('pool3', nn.MaxPool2d(kernel_size=2))
	model.add_module('conv4', nn.Conv2d(
		in_channels=128, out_channels=256, kernel_size=3, padding=1
	))
	model.add_module('relu4', nn.ReLU())
	model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
	model.add_module('flatten', nn.Flatten())
	model.add_module('fc1', nn.Linear(256, 1))
	model.add_module('sigmoid', nn.Sigmoid())

	return model


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):

	loss_metric = MeanMetric().to(device)
	acc_metric = BinaryAccuracy().to(device)
	f1_metric = BinaryF1Score().to(device)

	model.train()
	for x_batch, y_batch in tqdm(dataloader, desc='Train'):
		x_batch, y_batch = x_batch.to(device), y_batch.to(device)
		
		pred = model(x_batch)[:, 0] # [:, 0] ?
		loss = loss_fn(pred, y_batch.float()) # .float() ?

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		loss_metric.update(loss)
		acc_metric.update(pred, y_batch)
		f1_metric.update(pred, y_batch)

	epoch_loss = loss_metric.compute().item()
	epoch_acc = acc_metric.compute().item()
	epoch_f1 = f1_metric.compute().item()

	return epoch_loss, epoch_acc, epoch_f1

def evaluate(model, dataloader, loss_fn, device, desc='Validation'):
	model.eval()

	loss_metrics = MeanMetric().to(device)
	acc_metrics = BinaryAccuracy().to(device)
	f1_metrics = BinaryF1Score().to(device)	

	with torch.no_grad():
		for x_batch, y_batch in tqdm(dataloader, desc=desc):
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)
			pred = model(x_batch)
			y_batch = y_batch.unsqueeze(1).float()
			loss = loss_fn(pred, y_batch)

			loss_metrics.update(loss)
			acc_metrics.update(pred, y_batch)
			f1_metrics.update(pred, y_batch)
	
	return loss_metrics.compute().item(), acc_metrics.compute().item(),\
		   f1_metrics.compute().item()

def set_config():
	pass

def plot_metrics(metric_hist):
	epochs = np.arange(1, len(metric_hist['train_loss']) + 1)
	plt.figure(figsize=(12, 4))

	plt.subplot(1, 2, 1)
	plt.plot(epochs, metric_hist['train_loss'], '-o', label='Train Loss')
	plt.plot(epochs, metric_hist['valid_loss'], '--<', label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(epochs, metric_hist['train_acc'], '-o', label='Train Acc')
	plt.plot(epochs, metric_hist['valid_acc'], '--<', label='Validation Acc')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(config['plot_save_path'])
	print(f'Metric plot saved at \'{config["plot_save_path"]}\'\n')

	plt.show()

config = {
	'batch_size' : 128,
	'learning_rate' : 1e-3,
	'num_epochs' : 50,
	'best_model_path' : 'best_model.pth',
	'seed' : 1,
	'train_transform' : transforms.Compose([
		transforms.RandomCrop([178, 178]),
		transforms.RandomHorizontalFlip(),
		transforms.Resize([64, 64]),
		transforms.ToTensor(),
	]),
	'eval_transform' : transforms.Compose([
		transforms.CenterCrop([178, 178]),
	    transforms.Resize([64, 64]),
    	transforms.ToTensor(),
	]),
	'target_transform' : lambda attr: attr[31],
	'only_train' : False,
	'plot_save_path' : 'Metrics_plot.png'
}

def main():
	torch.manual_seed(config['seed'])
	device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

	train_transform = config['train_transform']
	eval_transform = config['eval_transform']
	target_transform = config['target_transform']

	train_dl, valid_dl, test_dl = get_dataloaders(
		batch_size=config['batch_size'],
		train_transform=train_transform,
		eval_transform=eval_transform,
		target_transform=target_transform
	)
	model = get_model().to(device)
	loss_fn = nn.BCELoss()
	only_test = config['only_train']
	
	if not only_test:
		print(f'### Config: num_epoch={config["num_epochs"]},\
		 	  batch size={config["batch_size"]},\
			  initial learning rate={config["learning_rate"]}\n')
		optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

		metric_hist = {
			'train_loss' : [],
			'valid_loss' : [],
			'train_acc' : [],
			'valid_acc' : [],
			'train_f1' : [],
			'valid_f1' : []
		}
		best_val_acc = 0

		print(f'Start Training (using device: {device})')
		for epoch in range(config['num_epochs']):
			print(f'\n### Epoch {epoch+1}/{config["num_epochs"]}')
			train_loss, train_acc, train_f1 = train_one_epoch(
				model, train_dl, loss_fn, optimizer, device)
			valid_loss, valid_acc, valid_f1 = evaluate(
				model, valid_dl, loss_fn, device
			)

			metric_hist['train_loss'].append(train_loss)
			metric_hist['valid_loss'].append(valid_loss)
			metric_hist['train_acc'].append(train_acc)
			metric_hist['valid_acc'].append(valid_acc)
			metric_hist['train_f1'].append(train_f1)
			metric_hist['valid_f1'].append(valid_f1)

			print(f'train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}')
			print(f'train acc: {train_acc:.4f}, valid acc: {valid_acc:.4f}')
			print(f'train f1 score: {train_f1:.4f}, valid f1 score: {valid_f1:.4f}')

			if valid_acc > best_val_acc:
				torch.save(model.state_dict(), config['best_model_path'])
				best_val_acc = valid_acc
				print(f'\n# Best model saved. Epoch:{epoch+1}, validation acc. {valid_acc:.4f}')
		print(f'Train finished...')
		plot_metrics(metric_hist)

	model.load_state_dict(torch.load(config['best_model_path']))
	print(f'model loaded from \'{config["best_model_path"]}\'')
	model.eval()

	print('### Start Test')
	_, test_acc, _ = evaluate(
		model, test_dl, loss_fn, device, desc='Test')
	print(f'Test Acc. {test_acc:.4f}')

if __name__ == '__main__':
	main()