from __future__ import print_function
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import Normalizer

# 從新的模組檔案中導入模型定義
from model.model_definitions import SimpleEncoder, SimSiam, simsiam_loss

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

# 2. 資料載入和預處理
traindata = pd.read_csv('dnn/kdd/binary/Training.csv', header=None)

X = traindata.iloc[:, 1:42]
trainX = np.array(X)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

X_train = torch.tensor(trainX, dtype=torch.float32).to(device)

# SimSiam 的訓練數據需要兩個增強視圖。對於表格數據，這意味著添加噪聲。
def get_augmented_views(data_tensor, noise_std=0.01):
    noise1 = torch.randn_like(data_tensor) * noise_std
    noise2 = torch.randn_like(data_tensor) * noise_std
    return data_tensor + noise1, data_tensor + noise2

# DataLoader for SimSiam
train_dataset = TensorDataset(X_train) # SimSiam 不需要標籤
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


# 3. 定義 SimSiam 模型
input_dim = 41
encoder_output_dim = 1024
projector_inner_dim = 32

encoder_model = SimpleEncoder(input_dim, encoder_output_dim).to(device)
simsiam_model = SimSiam(
    pretrained_model=encoder_model,
    encoder_output_dim=encoder_output_dim,
    projector_inner_dim=projector_inner_dim
).to(device)

# 4. 定義優化器
optimizer = optim.Adam(simsiam_model.parameters(), lr=0.0005)

# 5. 訓練循環和保存邏輯
epochs = 100
results_base_dir = "dnn/kddresults/dnn1layer"
# SimSiam通常只保存encoder，因為projector和predictor是訓練輔助
encoder_checkpoint_filepath = os.path.join(results_base_dir, "encoder-checkpoint-{epoch:02d}.pth")
encoder_model_save_path = os.path.join(results_base_dir, "simsiam_encoder_final.pth")

# 確保結果目錄存在
os.makedirs(results_base_dir, exist_ok=True)

# CSVLogger 記錄 SimSiam loss
csv_log_filepath = os.path.join(results_base_dir, "simsiam_training_analysis.csv")
csv_log_header = "epoch,loss\n"
csv_log_file = open(csv_log_filepath, 'w')
csv_log_file.write(csv_log_header)

best_loss = float('inf') # 用於 ModelCheckpoint (這裡基於 SimSiam loss)

print("\nStarting SimSiam pre-training...")
for epoch in range(epochs):
    simsiam_model.train()
    running_loss = 0.0
    total_samples = 0

    for i, (data_batch,) in enumerate(train_loader):
        data_batch = data_batch.to(device)
        x1, x2 = get_augmented_views(data_batch)

        optimizer.zero_grad()

        p1, p2, z1, z2 = simsiam_model(x1, x2)
        loss = simsiam_loss(p1, p2, z1, z2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data_batch.size(0)
        total_samples += data_batch.size(0)

    epoch_loss = running_loss / total_samples

    print(f"Epoch {epoch+1}/{epochs}, SimSiam Loss: {epoch_loss:.4f}")

    # ModelCheckpoint (保存最佳 encoder)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(simsiam_model.encoder.state_dict(), encoder_checkpoint_filepath.format(epoch=epoch+1))
        print(f"  --> Encoder checkpoint saved for epoch {epoch+1} (Loss: {epoch_loss:.4f})")

    # CSVLogger
    csv_log_file.write(f"{epoch+1},{epoch_loss:.4f}\n")
    csv_log_file.flush()

csv_log_file.close()

# 6. 保存最終模型 (通常只保存 encoder 用於下游任務)
torch.save(simsiam_model.encoder.state_dict(), encoder_model_save_path)
print(f"\nFinal Encoder model saved to {encoder_model_save_path}")

print("SimSiam pre-training finished.")