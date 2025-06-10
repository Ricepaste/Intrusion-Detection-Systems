from __future__ import print_function
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import Normalizer

# 設定隨機種子以確保可重現性
np.random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 檢查 GPU 是否可用並設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")


# 2. 資料載入 
traindata = pd.read_csv('dataset/NSL-KDD/KDDTrain+_20Percent.csv', header=0)
testdata = pd.read_csv('dataset/NSL-KDD/KDDTest+.csv', header=0)

X = traindata.iloc[:, 0:41]
Y = traindata.iloc[:, 41]
C = testdata.iloc[:, 41]
T = testdata.iloc[:, 0:41]

trainX = np.array(X)
testT = np.array(T)

# 注意: numpy 的 astype(float) 返回的是浮點數，但 PyTorch 需要更精確的類型，通常是 float32
# 這裡先保持 numpy 陣列類型，之後轉換為 PyTorch tensor 時指定 float32
# trainX.astype(float) # 這一行在 numpy 中是返回拷貝，不修改原陣列， PyTorch 轉換時會處理
# testT.astype(float)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)
testT = scaler.transform(testT)

y_train = np.array(Y)
y_test = np.array(C)

X_train_np = np.array(trainX)
X_test_np = np.array(testT)

# 將 NumPy 陣列轉換為 PyTorch 張量，並移動到指定設備
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device) # 對於 BCELoss，y_train 需要是 [batch_size, 1]
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# 創建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 使用相同的 batch_size

# 3. 定義 PyTorch 模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.dropout = nn.Dropout(0.01)
        self.ReLu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1) # 輸出層為 1 (二元分類)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x)) # 輸出層直接接 Sigmoid
        return x

input_dim = 41 # 輸入特徵數量

model = SimpleNN(input_dim).to(device) # 將模型移動到 GPU (如果可用)

# 4. 定義損失函數和優化器
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 優化器，這裡設定一個預設學習率

# 5. 訓練循環和回調函數 (手動實現)
epochs = 100
results_base_dir = "dnn/kddresults/dnn1layer"
checkpoint_filepath = os.path.join(results_base_dir, "NSLKDD_DNN_checkpoint-{epoch:02d}.pth") # PyTorch 習慣用 .pth
csv_log_filepath = os.path.join(results_base_dir, "NSLKDD_DNN_training_set_dnnanalysis.csv")
model_save_path = os.path.join(results_base_dir, "NSLKDD_DNN_dnn1layer_model.pth")

# 確保結果目錄存在
os.makedirs(results_base_dir, exist_ok=True)

best_loss = float('inf') # 用於 ModelCheckpoint
csv_log_header = "epoch,loss,accuracy\n"
csv_log_file = open(csv_log_filepath, 'w')
csv_log_file.write(csv_log_header)

print("\nStarting training...")
for epoch in range(epochs):
    model.train() # 設定模型為訓練模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad() # 清除梯度

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # 反向傳播
        optimizer.step() # 更新權重

        running_loss += loss.item() * inputs.size(0) # 累計損失
        predicted = (outputs > 0.5).float() # 二元分類預測 (閾值 0.5)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # ModelCheckpoint (保存最佳模型)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), checkpoint_filepath.format(epoch=epoch+1)) # 只保存模型參數
        print(f"  --> Checkpoint saved for epoch {epoch+1} (Loss: {epoch_loss:.4f})")

    # CSVLogger
    csv_log_file.write(f"{epoch+1},{epoch_loss:.4f},{epoch_accuracy:.4f}\n")
    csv_log_file.flush() # 確保寫入檔案

# 訓練結束後關閉 CSV log 檔案
csv_log_file.close()

# 6. 保存最終模型
# PyTorch 通常保存模型的 state_dict (參數)
torch.save(model.state_dict(), model_save_path)
print(f"\nFinal model saved to {model_save_path}")

# (可選) 載入模型並在測試集上評估
model.load_state_dict(torch.load(model_save_path))
model.eval() # 設定模型為評估模式
with torch.no_grad():
    test_outputs = model(X_test)
    test_predicted = (test_outputs > 0.5).float()
    test_accuracy = accuracy_score(y_test.cpu().numpy(), test_predicted.cpu().numpy())
    print(f"Test Accuracy: {test_accuracy:.4f}")