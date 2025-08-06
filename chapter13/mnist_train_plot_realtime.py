import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from torchmetrics import Accuracy
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MultiLayerPerceptron(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()
        
        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        all_layers = [nn.Flatten()] # (1, 28, 28) → (1, 784) (2D → 1D Linear) 
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
        
        all_layers.append(nn.Linear(hidden_units[-1], 10))
        self.model = nn.Sequential(*all_layers)
        
    def forward(self, x):
        return self.model(x)

def load_and_prepare_data(data_path='./'):
    transform = transforms.Compose([transforms.ToTensor()])
    MNIST(root=data_path, download=True)

    mnist_all = MNIST(root=data_path, train=True, transform=transform, download=False)
    train_set, val_set = random_split(
        mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(1)
    )
    test_set = MNIST(root=data_path, train=False, transform=transform, download=False)

    train_loader = DataLoader(train_set, batch_size=64, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=2)

    return train_loader, val_loader, test_loader

def update_graphs(fig, ax1, ax2, train_loss_line, val_loss_line, train_acc_line, val_acc_line, 
                  train_losses, val_losses, train_accuracies, val_accuracies):
    
    epochs_ran = len(train_losses)
    
    # Loss 그래프 업데이트
    ax1.set_xlim(0, epochs_ran + 1)
    current_max_loss = max(max(train_losses) if train_losses else 0.1,
                           max(val_losses) if val_losses else 0.1)
    ax1.set_ylim(0, current_max_loss * 1.1)
    train_loss_line.set_data(range(1, epochs_ran + 1), train_losses)
    val_loss_line.set_data(range(1, epochs_ran + 1), val_losses)
    
    # Accuracy 그래프 업데이트
    ax2.set_xlim(0, epochs_ran + 1)
    ax2.set_ylim(0, 1.05)
    train_acc_line.set_data(range(1, epochs_ran + 1), train_accuracies)
    val_acc_line.set_data(range(1, epochs_ran + 1), val_accuracies)

    # Figure 캔버스 업데이트
    fig.canvas.draw_idle()
    plt.pause(0.01)


def main():
    torch.manual_seed(1)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    device = 'cpu'
    print(f'Using device: {device}')

    model = MultiLayerPerceptron().to(device)
    train_loader, val_loader, test_loader = load_and_prepare_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    train_acc_metric = Accuracy(task='multiclass', num_classes=10).to(device)
    val_acc_metric = Accuracy(task='multiclass', num_classes=10).to(device)
    test_acc_metric = Accuracy(task='multiclass', num_classes=10).to(device)

    epochs = 20
    best_val_acc = 0.0
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training and Validation Metrics', fontsize=16)

    # 손실 그래프 (왼쪽)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    train_loss_line, = ax1.plot([], [], label='Train Loss', color='blue', marker='o')
    val_loss_line, = ax1.plot([], [], label='Validation Loss', color='orange', marker='s')
    ax1.legend()

    # 정확도 그래프 (오른쪽)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    train_acc_line, = ax2.plot([], [], label='Train Accuracy', color='blue', marker='o')
    val_acc_line, = ax2.plot([], [], label='Validation Accuracy', color='orange', marker='s')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 그래프 간격 조절
    
    plt.ion()
    plt.show()
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        model.train()
        running_train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(tqdm(
            train_loader, desc='Training')):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            train_acc_metric.update(preds, labels)
            running_train_loss += loss.item() * images.size(0)
        
        train_acc = train_acc_metric.compute()
        
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc.item())
        
        print(f'Train Loss: {loss.item():.4f}, Train Acc: {train_acc.item():.4f}')
        train_acc_metric.reset()
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(
                val_loader, desc='Validation'
            )):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss = loss_fn(logits, labels)
                running_val_loss += val_loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                val_acc_metric.update(preds, labels)
        
        val_acc = val_acc_metric.compute()
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc.item())
        
        print(f'Validation Acc: {val_acc.item():.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'New best model saved with validation accuracy: {best_val_acc.item():.4f}')
        
        val_acc_metric.reset()
        
        update_graphs(fig, ax1, ax2, train_loss_line, val_loss_line, train_acc_line, val_acc_line,
                      train_losses, val_losses, train_accuracies, val_accuracies)
            
    save_path = 'final_combined_metrics_plot.png'
    fig.savefig(save_path)
    print(f'Final combined metrics plot saved to {save_path}')

    plt.ioff()
    plt.show()
    
    print('\n--- Testing best model ---')

    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        best_model = MultiLayerPerceptron().to(device)
        best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        best_model.eval()
    else:
        print(f'Warning: No best model checkpoint found. Using the last trained model.')
        best_model = model
        best_model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc='Testing')):
            images, labels = images.to(device), labels.to(device)
            
            logits = best_model(images)
            preds = torch.argmax(logits, dim=1)
            test_acc_metric.update(preds, labels)
    
    test_acc = test_acc_metric.compute()
    print(f'Final Test Accuracy: {test_acc.item():.4f}')

if __name__ == '__main__':
    main()