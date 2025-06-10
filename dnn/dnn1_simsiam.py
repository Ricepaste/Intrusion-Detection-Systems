from __future__ import print_function
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # 用於餘弦相似度
from torch.utils.data import TensorDataset, DataLoader
from model.Model import SimpleEncoder, SimSiam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# --- 從這裡開始包含上面定義的 SimpleEncoder 和 SimSiam 類 ---
# 請確保將 SimpleEncoder 和 SimSiam 的類定義複製到你的腳本中
# -------------------------------------------------------------

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
# testdata = pd.read_csv('dnn/kdd/binary/Testing.csv', header=None) # SimSiam 預訓練通常只用無標籤數據

X = traindata.iloc[:, 1:42]
# Y = traindata.iloc[:, 0] # SimSiam 預訓練不使用標籤 Y

trainX = np.array(X)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

X_train_np = np.array(trainX)

# 為了 SimSiam 預訓練，我們只關注 X_train
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)

# SimSiam 的訓練數據需要兩個增強視圖。對於表格數據，這可能意味著添加噪聲。
# 這裡我們生成兩個帶有隨機噪聲的視圖
def get_augmented_views(data_tensor, noise_std=0.01):
    noise1 = torch.randn_like(data_tensor) * noise_std
    noise2 = torch.randn_like(data_tensor) * noise_std
    return data_tensor + noise1, data_tensor + noise2

# DataLoader for SimSiam
train_dataset = TensorDataset(X_train) # SimSiam 不需要標籤
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


# 3. 定義 SimSiam 模型
input_dim = 41 # 原始輸入特徵數量
encoder_output_dim = 1024 # SimpleEncoder 的輸出維度，與你的原始模型一致
projector_inner_dim = 32 # 投影器中間層維度，可調整

encoder_model = SimpleEncoder(input_dim, encoder_output_dim).to(device)
simsiam_model = SimSiam(
    pretrained_model=encoder_model,
    encoder_output_dim=encoder_output_dim,
    projector_inner_dim=projector_inner_dim
).to(device)

# 4. 定義 SimSiam 的損失函數 (負餘弦相似度) 和優化器
# SimSiam loss: D(p1, z2) + D(p2, z1) where D(p, z) = -cosine_similarity(p, z)
def simsiam_loss(p1, p2, z1, z2):
    # D(p1, z2)
    loss_p1_z2 = -F.cosine_similarity(p1, z2, dim=-1).mean()
    # D(p2, z1)
    loss_p2_z1 = -F.cosine_similarity(p2, z1, dim=-1).mean()
    return (loss_p1_z2 + loss_p2_z1) / 2

optimizer = optim.Adam(simsiam_model.parameters(), lr=0.0005) # Adam 優化器，學習率可能需要調整

# 5. 訓練循環和保存邏輯
epochs = 100
results_base_dir = "dnn/kddresults/dnn1layer"
# SimSiam通常只保存encoder，因為projector和predictor是訓練輔助
encoder_checkpoint_filepath = os.path.join(results_base_dir, "encoder-checkpoint-{epoch:02d}.pth")
simsiam_model_save_path = os.path.join(results_base_dir, "simsiam_model.pth")
encoder_model_save_path = os.path.join(results_base_dir, "simsiam_encoder_final.pth")

# 確保結果目錄存在
os.makedirs(results_base_dir, exist_ok=True)

# CSVLogger 記錄 SimSiam loss
csv_log_filepath = os.path.join(results_base_dir, "simsiam_training_analysis.csv")
csv_log_header = "epoch,loss\n" # SimSiam 通常只關注 loss
csv_log_file = open(csv_log_filepath, 'w')
csv_log_file.write(csv_log_header)

best_loss = float('inf') # 用於 ModelCheckpoint (這裡基於 SimSiam loss)

print("\nStarting SimSiam pre-training...")
for epoch in range(epochs):
    simsiam_model.train() # 設定模型為訓練模式
    running_loss = 0.0
    total_samples = 0

    for i, (data_batch,) in enumerate(train_loader): # DataLoader now yields (data_tensor,)
        data_batch = data_batch.to(device)
        x1, x2 = get_augmented_views(data_batch) # 生成兩個增強視圖

        optimizer.zero_grad() # 清除梯度

        p1, p2, z1, z2 = simsiam_model(x1, x2)
        loss = simsiam_loss(p1, p2, z1, z2) # 計算 SimSiam 損失
        loss.backward() # 反向傳播
        optimizer.step() # 更新權重

        running_loss += loss.item() * data_batch.size(0)
        total_samples += data_batch.size(0)

    epoch_loss = running_loss / total_samples

    print(f"Epoch {epoch+1}/{epochs}, SimSiam Loss: {epoch_loss:.4f}")

    # ModelCheckpoint (保存最佳 encoder)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        # 保存 encoder 的狀態字典
        torch.save(simsiam_model.encoder.state_dict(), encoder_checkpoint_filepath.format(epoch=epoch+1))
        print(f"  --> Encoder checkpoint saved for epoch {epoch+1} (Loss: {epoch_loss:.4f})")

    # CSVLogger
    csv_log_file.write(f"{epoch+1},{epoch_loss:.4f}\n")
    csv_log_file.flush()

# 訓練結束後關閉 CSV log 檔案
csv_log_file.close()

# 6. 保存最終模型 (通常只保存 encoder 用於下游任務)
torch.save(simsiam_model.state_dict(), simsiam_model_save_path) # 保存整個 SimSiam 模型的參數 (包含 projector 和 predictor)
torch.save(simsiam_model.encoder.state_dict(), encoder_model_save_path) # 單獨保存 encoder 參數
print(f"\nFinal SimSiam model saved to {simsiam_model_save_path}")
print(f"Final Encoder model saved to {encoder_model_save_path}")

print("SimSiam pre-training finished.")

# --- 接下來，你可以使用保存的 encoder_model_save_path 來載入預訓練的 encoder，並為你的下游分類任務構建新的模型 ---
# 示例：如何載入預訓練的 encoder
loaded_encoder = SimpleEncoder(input_dim, encoder_output_dim).to(device)
loaded_encoder.load_state_dict(torch.load(encoder_model_save_path))
loaded_encoder.eval() # 設定為評估模式

# 如果需要進行分類，你會在預訓練的 encoder 後面添加一個分類頭
class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(DownstreamClassifier, self).__init__()
        self.encoder = encoder
        # 凍結 encoder 參數 (通常預訓練後不會再訓練 encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(encoder_output_dim, num_classes) # 分類頭

    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return output

# 假設 num_classes 是你的二元分類 (2 個類別)
num_classes = 2
classifier_model = DownstreamClassifier(loaded_encoder, num_classes).to(device)
# 然後用你的 y_train 數據訓練這個 classifier_model