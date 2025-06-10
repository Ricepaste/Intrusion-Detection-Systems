from __future__ import print_function
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 從新的模組檔案中導入模型定義
from model.model_definitions import SimpleEncoder, DownstreamClassifier

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

# 2. 資料載入和預處理 (訓練和測試數據)
traindata = pd.read_csv('dataset/NSL-KDD/KDDTrain+_20Percent.csv', header=0)
testdata = pd.read_csv('dataset/NSL-KDD/KDDTest+.csv', header=0)

# 訓練集
X_train_data = traindata.iloc[:, 0:41]
y_train_data = traindata.iloc[:, 41]

# 測試集
X_test_data = testdata.iloc[:, 0:41]
y_test_data = testdata.iloc[:, 41] # 這裡需要測試集的標籤 C

# 將 NumPy 陣列轉換為浮點數類型
trainX_np = np.array(X_train_data).astype(np.float32)
testT_np = np.array(X_test_data).astype(np.float32)
y_train_np = np.array(y_train_data).astype(np.float32)
y_test_np = np.array(y_test_data).astype(np.float32)


# 歸一化
scaler_train = Normalizer().fit(trainX_np)
trainX_normalized = scaler_train.transform(trainX_np)

scaler_test = Normalizer().fit(testT_np)
testT_normalized = scaler_test.transform(testT_np)


# 轉換為 PyTorch 張量並移動到設備
X_train = torch.tensor(trainX_normalized, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
X_test = torch.tensor(testT_normalized, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

# DataLoader for Downstream Task
batch_size = 64 # 分類任務的 batch_size
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. 載入預訓練的 Encoder 模型
input_dim = 41
encoder_output_dim = 1024 # 確保與預訓練時的 encoder_output_dim 一致

encoder_model_path = "dnn/kddresults/dnn1layer/NSLKDD_simsiam_encoder_final.pth" # 預訓練模型保存路徑
results_base_dir = "dnn/kddresults/dnn1layer"

# 檢查預訓練模型是否存在
if not os.path.exists(encoder_model_path):
    print(f"Error: Pre-trained encoder model not found at {encoder_model_path}")
    print("Please run simsiam_pretrain.py first to train the encoder.")
    exit()

# 實例化 Encoder 並載入預訓練權重
encoder_model = SimpleEncoder(input_dim, encoder_output_dim).to(device)
encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=device))
# 設定為評估模式，但在 DownstreamClassifier 中會再次處理凍結
encoder_model.eval()

# 4. 定義下游分類器
num_classes = 1 # 二元分類，輸出為 1 (經 Sigmoid 後為 0 或 1)
classifier_model = DownstreamClassifier(encoder=encoder_model,
                                        encoder_output_dim=encoder_output_dim,
                                        num_classes=num_classes).to(device)

# 確保 encoder 參數被凍結 (在 DownstreamClassifier 中已設置 requires_grad=False)
# 只有 classifier 層的參數將會被訓練
# optimizer_classifier = optim.Adam(classifier_model.classifier.parameters(), lr=0.001) # 只優化分類頭
optimizer_classifier = optim.Adam(classifier_model.parameters(), lr=0.001) # 全優化
criterion_classifier = nn.BCEWithLogitsLoss() # 結合 Sigmoid 和 BCELoss，更穩定


# 5. 下游任務訓練循環
downstream_epochs = 50 # 下游任務的訓練 epochs
print("\nStarting Downstream Classification training...")

for epoch in range(downstream_epochs):
    encoder_model.train()
    classifier_model.train() # 設定模型為訓練模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_classifier.zero_grad()

        outputs = classifier_model(inputs)
        loss = criterion_classifier(outputs, labels) # 使用 BCEWithLogitsLoss

        loss.backward()
        optimizer_classifier.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float() # BCEWithLogitsLoss 的輸出沒有 Sigmoid，需要手動應用
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    print(f"Downstream Epoch {epoch+1}/{downstream_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# 6. 下游任務評估
print("\nStarting Downstream Classification evaluation...")
encoder_model.eval()
classifier_model.eval() # 設定模型為評估模式 (禁用 dropout 等)
all_predictions = []
all_true_labels = []

with torch.no_grad(): # 在評估時禁用梯度計算
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = classifier_model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

# 計算評估指標
all_predictions = np.array(all_predictions).flatten()
all_true_labels = np.array(all_true_labels).flatten()

accuracy = accuracy_score(all_true_labels, all_predictions)
precision = precision_score(all_true_labels, all_predictions, zero_division=0)
recall = recall_score(all_true_labels, all_predictions, zero_division=0)
f1 = f1_score(all_true_labels, all_predictions, zero_division=0)
cm = confusion_matrix(all_true_labels, all_predictions)

print("\n--- Downstream Classification Results ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
print("---------------------------------------")

# 7. 保存最終分類器模型 (可選)
classifier_model_save_path = os.path.join(results_base_dir, r"NSLKDD_downstream_classifier_final.pth")
torch.save(classifier_model.state_dict(), classifier_model_save_path)
print(f"\nFinal Downstream Classifier model saved to {classifier_model_save_path}")