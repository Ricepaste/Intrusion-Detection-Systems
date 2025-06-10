from __future__ import print_function
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import Normalizer, StandardScaler 

from model.model_definitions import SimpleEncoder, SimSiam, simsiam_loss

# ==== Dataset 定義 ====
class UNSWDataset(Dataset):
    def __init__(self, csv_file, feature_cols, label_col, normalizer=None):
        df = pd.read_csv(csv_file)
        X = df.iloc[:, feature_cols].values.astype(np.float32)

        if normalizer is not None:
            X = normalizer.transform(X)

        self.X = torch.from_numpy(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# ==== 基本設定 ====
np.random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==== 資料讀取與標準化 ====
train_df = pd.read_csv('./dataset/UNSW_NB15/Multiclass/UNSW_NB15_training_multiclass.csv')
feature_indices = list(range(1, train_df.shape[1]-1))

# normalizer = Normalizer()
normalizer = StandardScaler()
normalizer.fit(train_df.iloc[:, feature_indices].values.astype(np.float32))

train_dataset = UNSWDataset(
    csv_file='./dataset/UNSW_NB15/Multiclass/UNSW_NB15_training_multiclass.csv',
    feature_cols=feature_indices,
    label_col=-1,
    normalizer=normalizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

input_dim = len(feature_indices)

# ==== 定義 SimSiam 模型 ====
encoder_output_dim = 1024
projector_inner_dim = 512

encoder_model = SimpleEncoder(input_dim, encoder_output_dim).to(device)
simsiam_model = SimSiam(
    pretrained_model=encoder_model,
    encoder_output_dim=encoder_output_dim,
    projector_inner_dim=projector_inner_dim
).to(device)

optimizer = optim.Adam(simsiam_model.parameters(), lr=0.0005)

# ==== 資料增強 ====
# def get_augmented_views(data_tensor, noise_std=0.00001):
#     noise1 = torch.randn_like(data_tensor) * noise_std
#     noise2 = torch.randn_like(data_tensor) * noise_std
#     return data_tensor + noise1, data_tensor + noise2

def get_augmented_views(data_tensor, noise_factor=0.05):
    """
    為輸入的 data_tensor 生成兩個增強視圖。
    噪音的標準差是根據每個特徵自身的標準差進行縮放的。

    參數:
    data_tensor (torch.Tensor): 輸入數據張量，形狀為 (batch_size, num_features)。
                                 例如 (1024, 42)。
    noise_factor (float): 一個縮放因子，用於調整添加到每個特徵的噪音量。
                           噪音的實際標準差將是 feature_std * noise_factor。

    返回:
    tuple: 包含兩個增強視圖 (augmented_view1, augmented_view2) 的元組。
    """
    # 檢查 data_tensor 的維度
    if data_tensor.ndim != 2:
        raise ValueError(f"data_tensor 必須是2維的 (batch_size, num_features)，但得到的是 {data_tensor.ndim} 維")

    # 1. 計算每個特徵（每一列）的標準差
    feature_stds = torch.std(data_tensor, dim=0, keepdim=True, unbiased=True)

    # 2. 計算每個特徵應添加的噪音的目標標準差
    target_noise_std_per_feature = feature_stds * noise_factor

    # 3. 生成兩個獨立的標準正態分佈噪音張量
    base_noise1 = torch.randn_like(data_tensor)
    base_noise2 = torch.randn_like(data_tensor)

    # 4. 根據每個特徵的 target_noise_std_per_feature 來縮放噪音
    scaled_noise1 = base_noise1 * target_noise_std_per_feature
    scaled_noise2 = base_noise2 * target_noise_std_per_feature

    # 5. 生成增強視圖
    augmented_view1 = data_tensor + scaled_noise1
    augmented_view2 = data_tensor + scaled_noise2

    return augmented_view1, augmented_view2

# ==== 訓練邏輯 ====
epochs = 200
results_base_dir = "dnn/kddresults/dnn1layer"
encoder_checkpoint_filepath = os.path.join(results_base_dir, "encoder-checkpoint-{epoch:02d}.pth")
encoder_model_save_path = os.path.join(results_base_dir, "simsiam_encoder_final.pth")

os.makedirs(results_base_dir, exist_ok=True)
csv_log_filepath = os.path.join(results_base_dir, "UNSW_NB15_simsiam_training_analysis.csv")
csv_log_file = open(csv_log_filepath, 'w')
csv_log_file.write("epoch,loss\n")

best_loss = float('inf')

print("\nStarting SimSiam pre-training...")
for epoch in range(epochs):
    simsiam_model.train()
    running_loss = 0.0
    total_samples = 0

    for data_batch in train_loader:
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

    # ModelCheckpoint
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(simsiam_model.encoder.state_dict(), encoder_checkpoint_filepath.format(epoch=epoch+1))
        print(f"  --> Encoder checkpoint saved for epoch {epoch+1} (Loss: {epoch_loss:.4f})")

    csv_log_file.write(f"{epoch+1},{epoch_loss:.4f}\n")
    csv_log_file.flush()

csv_log_file.close()

# 保存最終 encoder
torch.save(simsiam_model.encoder.state_dict(), encoder_model_save_path)
print(f"\nFinal Encoder model saved to {encoder_model_save_path}")
print("SimSiam pre-training finished.")
