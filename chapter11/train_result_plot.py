import pandas as pd
import matplotlib.pyplot as plt

file_path = 'train_result/loss_acc_result_no_index.csv'
loaded_df = pd.read_csv(file_path)

loss_np = loaded_df['epoch_loss'].to_numpy()
train_acc_np = loaded_df['epoch_train_acc'].to_numpy()
valid_acc_np = loaded_df['epoch_valid_acc'].to_numpy()

plt.plot(range(len(loss_np)), loss_np)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.show()

plt.plot(range(len(train_acc_np)), train_acc_np, label='Training')
plt.plot(range(len(valid_acc_np)), valid_acc_np, label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()