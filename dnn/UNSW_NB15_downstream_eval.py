from __future__ import print_function
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import Normalizer # 你使用的是 Normalizer
from sklearn.preprocessing import StandardScaler # 也可以考慮 StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 從新的模組檔案中導入模型定義
from model.model_definitions import SimpleEncoder, DownstreamClassifier

# ==== START: 引入 UNSWDataset 類別定義 (修改後) ====
class UNSWDataset(Dataset):
    def __init__(self, csv_file, feature_cols, label_col, normalizer=None,
                 label_mapping_ref=None, num_classes_ref=None, dataset_name=""): # 新增參數
        """
        Args:
            csv_file (str): Path to the CSV file.
            feature_cols (list of int): 欲取用的欄位 index 或名稱。
            label_col (int or str): 標籤欄位的 index 或名稱。
            normalizer (sklearn Normalizer or StandardScaler): 如果提供則對 X 做正規化/標準化。
            label_mapping_ref (dict, optional): 預先定義的標籤映射。供測試集/驗證集使用。
            num_classes_ref (int, optional): 預先定義的類別數量。供測試集/驗證集使用。
            dataset_name (str, optional): 用於打印信息的數據集名稱。
        """
        df = pd.read_csv(csv_file,header=0)
        X_data = df.iloc[:, feature_cols].values.astype(np.float32)
        y_raw = df.iloc[:, label_col].values

        if label_mapping_ref is None or num_classes_ref is None:
            # 學習標籤映射 (通常在訓練集上)
            unique_labels_from_file = np.unique(y_raw)
            self.label_mapping = {label: i for i, label in enumerate(unique_labels_from_file)}
            self.num_classes = len(unique_labels_from_file)
        else:
            # 使用提供的標籤映射 (通常在測試集/驗證集上)
            self.label_mapping = label_mapping_ref
            self.num_classes = num_classes_ref

        # 使用 self.label_mapping 將 y_raw 映射到整數索引
        y_mapped_intermediate = []
        unknown_labels_found = set()
        for label_val in y_raw:
            mapped_val = self.label_mapping.get(label_val, -1) # -1 表示未知標籤
            if mapped_val == -1:
                unknown_labels_found.add(label_val)
            y_mapped_intermediate.append(mapped_val)
        y_mapped_intermediate = np.array(y_mapped_intermediate)

        if unknown_labels_found:
            print(f"Warning for [{dataset_name}]: The following labels were found in the file but not in the reference label_mapping: {list(unknown_labels_found)}. Samples with these labels will be excluded.")

        # 過濾掉那些在訓練集 mapping 中不存在的標籤 (-1) 的樣本
        valid_indices = y_mapped_intermediate != -1
        if not np.all(valid_indices):
            original_count = len(y_mapped_intermediate)
            X_data = X_data[valid_indices]
            y_final_mapped = y_mapped_intermediate[valid_indices]
            print(f"[{dataset_name}] Filtered out {original_count - len(y_final_mapped)} samples due to unknown labels.")
            if len(y_final_mapped) == 0:
                raise ValueError(f"No valid samples remaining in [{dataset_name}] after filtering unknown labels. Check label consistency or data.")
        else:
            y_final_mapped = y_mapped_intermediate

        y = y_final_mapped.astype(np.int64) # CrossEntropyLoss 期望 int64 類別索引

        # 正規化/標準化
        if normalizer is not None:
            # 假設 normalizer 的 fit 操作已在外部對訓練數據完成
            # 對於測試集，只進行 transform
            X_data = normalizer.transform(X_data)

        self.X = torch.from_numpy(X_data)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# ==== END: 引入 UNSWDataset 類別定義 ====


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

# 2. 資料載入和預處理 (訓練和測試數據)
# ==== START: 使用 UNSWDataset 和 DataLoader 載入資料 (修改後) ====
# 定義資料路徑
TRAIN_CSV_PATH = './dataset/UNSW_NB15/Multiclass/UNSW_NB15_training_multiclass.csv'
TEST_CSV_PATH = './dataset/UNSW_NB15/Multiclass/UNSW_NB15_testing_multiclass.csv'

# 讀取訓練資料集以確定特徵欄位和執行正規化 fitting
train_df_for_fitting = pd.read_csv(TRAIN_CSV_PATH, header=0)
# 假設特徵欄位從第二列到倒數第二列，最後一列是標籤
feature_indices = list(range(1, train_df_for_fitting.shape[1]-1))
label_col_index = train_df_for_fitting.shape[1]-1 # 假設最後一列是類別標籤 ('attack_cat' 或類似)


# 初始化並 fitting Normalizer 在訓練集特徵上
features_for_fitting = train_df_for_fitting.iloc[:, feature_indices].values.astype(np.float32)
normalizer = StandardScaler() # 或 StandardScaler()
normalizer.fit(features_for_fitting)
print(f"Normalizer/Scaler fitted on training data features.")

# 建立訓練資料集
train_dataset = UNSWDataset(
    csv_file=TRAIN_CSV_PATH,
    feature_cols=feature_indices,
    label_col=label_col_index,
    normalizer=normalizer, # 將已 fitting 的 normalizer 傳入
    dataset_name="TrainingSet"
)

# 從訓練數據集獲取學習到的 label_mapping 和 num_classes
learned_label_mapping = train_dataset.label_mapping
num_classes = train_dataset.num_classes # <--- 全局的 num_classes 來自訓練集

# 建立測試資料集，並傳入訓練集學到的 mapping 和 num_classes
test_dataset = UNSWDataset(
    csv_file=TEST_CSV_PATH,
    feature_cols=feature_indices,
    label_col=label_col_index,
    normalizer=normalizer, # 測試集也使用訓練集 fitting 的 normalizer
    label_mapping_ref=learned_label_mapping, # <--- 傳入訓練集的 mapping
    num_classes_ref=num_classes,             # <--- 傳入訓練集的 num_classes
    dataset_name="TestSet"
)

# 動態獲取輸入特徵維度 (從實際處理後的數據獲取，更安全)
input_dim = train_dataset.X.shape[1]
print(f"Detected input dimension for the model (from processed training data): {input_dim}")

# ==== END: 使用 UNSWDataset 和 DataLoader 載入資料 ====

# 檢查數據集是否為空
if len(train_dataset) == 0:
    raise ValueError("Training dataset is empty. Check data paths and label mapping.")
if len(test_dataset) == 0:
    print("Warning: Test dataset is empty. Evaluation will not be possible. Check test data path, labels, and mapping consistency with training set.")
    # 根據情況，你可能想在此處 exit() 或繼續（如果只是想測試訓練部分）

# 宣告 DataLoader
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0, # 根據你的系統調整，Windows下通常設0
    pin_memory=True if device.type == 'cuda' else False # pin_memory 只在 CUDA 上有效
)

if len(test_dataset) > 0:
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
else:
    test_loader = None # 如果測試集為空，則不創建 test_loader

# 3. 載入預訓練的 Encoder 模型
encoder_output_dim = 1024 # 確保與預訓練時的 encoder_output_dim 一致

encoder_model_path = "dnn/kddresults/dnn1layer/encoder-checkpoint-197.pth" # 預訓練模型保存路徑
results_base_dir = "dnn/kddresults/dnn1layer"
os.makedirs(results_base_dir, exist_ok=True) # 確保結果目錄存在

# 檢查預訓練模型是否存在
if not os.path.exists(encoder_model_path):
    print(f"Error: Pre-trained encoder model not found at {encoder_model_path}")
    print("Please run simsiam_pretrain.py first to train the encoder, or ensure the path is correct.")
    exit()

# 實例化 Encoder 並載入預訓練權重
encoder_model = SimpleEncoder(input_dim, encoder_output_dim).to(device)
try:
    encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=device))
    print(f"Successfully loaded pre-trained encoder from {encoder_model_path}")
except Exception as e:
    print(f"Error loading pre-trained encoder model: {e}")
    print("Please check if the encoder architecture matches the saved weights and if input_dim is correct.")
    exit()
encoder_model.eval() # 設置為評估模式，因為我們只用它來提取特徵

# 4. 定義下游分類器
# num_classes 是從訓練集的 UNSWDataset 中動態獲取的
classifier_model = DownstreamClassifier(encoder=encoder_model, # 傳入預訓練的 encoder
                                        encoder_output_dim=encoder_output_dim,
                                        num_classes=num_classes).to(device)

# 凍結 Encoder 的權重 (可選，但常見做法)
# 如果你想微調 (fine-tune) encoder，則不要執行這一步
# for param in classifier_model.encoder.parameters():
# param.requires_grad = False
# print("Froze encoder parameters for downstream task.")

# 只優化分類器的參數
optimizer_classifier = optim.Adam(classifier_model.parameters(), lr=0.001)
criterion_classifier = nn.CrossEntropyLoss()

# 5. 下游任務訓練循環
downstream_epochs = 100
print(f"\nStarting Downstream Classification training for {downstream_epochs} epochs...")

for epoch in range(downstream_epochs):
    classifier_model.train() # 確保模型在訓練模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples_epoch = 0 # 重命名以避免與 DataLoader 內部變量衝突

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_classifier.zero_grad()
        outputs = classifier_model(inputs)
        loss = criterion_classifier(outputs, labels)
        loss.backward()
        optimizer_classifier.step()
        
        # temp = outputs.detach().cpu().numpy()
        # temp_labels = labels.detach().cpu().numpy()[0]
        
        # # list temp 
        # temp = list(map(float,list(temp[0])))
        # temp = [round(x, 4) for x in temp] # 四捨五入到小數點後四位
        
        # print(f"Output: {temp} \n Labels: {temp_labels}")

        running_loss += loss.item() * inputs.size(0)
        predicted = torch.argmax(outputs, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples_epoch += labels.size(0)

    if total_samples_epoch > 0:
        epoch_loss = running_loss / total_samples_epoch
        epoch_accuracy = correct_predictions / total_samples_epoch
        print(f"Downstream Epoch {epoch+1}/{downstream_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    else:
        print(f"Downstream Epoch {epoch+1}/{downstream_epochs}, No samples in training loader for this epoch.")


# 6. 下游任務評估
if test_loader is not None and len(test_dataset) > 0 :
    print("\nStarting Downstream Classification evaluation...")
    classifier_model.eval() # 確保模型在評估模式
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier_model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 計算評估指標
    # flatten() 在這裡通常是多餘的，因為 predicted 和 labels 應該已經是1D的了
    all_predictions_np = np.array(all_predictions)
    all_true_labels_np = np.array(all_true_labels)

    if len(all_true_labels_np) > 0 and len(all_predictions_np) > 0:
        accuracy = accuracy_score(all_true_labels_np, all_predictions_np)
        # 使用 labels=np.arange(num_classes) 確保混淆矩陣考慮所有可能的類別，即使某些類別在測試集中沒有樣本或沒有被預測到
        # 這對 precision/recall/f1 score 的 'macro' average 尤其重要
        unique_true_labels_in_test = np.unique(all_true_labels_np)
        report_labels = np.arange(num_classes) # 考慮所有訓練時定義的類別
        
        # 更新: 確保 confusion_matrix 的 labels 參數只包含實際存在的標籤，或者所有可能的標籤
        # sklearn 0.24+ 版本，如果 labels 參數包含預測或真實值中沒有的標籤，會給出警告。
        # 為了得到完整的混淆矩陣 (num_classes x num_classes)，使用 report_labels
        cm_labels = np.unique(np.concatenate((all_true_labels_np, all_predictions_np))) # 實際出現的標籤
        # 如果想強制混淆矩陣大小為 num_classes x num_classes:
        # cm_labels_full = np.arange(num_classes)
        # cm = confusion_matrix(all_true_labels_np, all_predictions_np, labels=cm_labels_full)

        precision = precision_score(all_true_labels_np, all_predictions_np, average='weighted', zero_division=0, labels=report_labels)
        recall = recall_score(all_true_labels_np, all_predictions_np, average='weighted', zero_division=0, labels=report_labels)
        f1 = f1_score(all_true_labels_np, all_predictions_np, average='weighted', zero_division=0, labels=report_labels)
        cm = confusion_matrix(all_true_labels_np, all_predictions_np, labels=report_labels) # 使用 report_labels 確保維度

        print("\n--- Downstream Classification Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print("Confusion Matrix (rows: true, cols: predicted):")
        # 打印混淆矩陣時，可以加上標籤名稱以提高可讀性
        # target_names = [str(k) for k,v in sorted(learned_label_mapping.items(), key=lambda item: item[1])] # 從 mapping 獲取類別名
        # print(pd.DataFrame(cm, index=target_names, columns=target_names))
        print(cm)

        # 也可以計算 macro F1-score，它對類別不平衡更敏感
        f1_macro = f1_score(all_true_labels_np, all_predictions_np, average='macro', zero_division=0, labels=report_labels)
        print(f"F1-Score (macro): {f1_macro:.4f}")
        print("---------------------------------------")
    else:
        print("Evaluation could not be performed: No predictions or true labels collected from the test set.")
else:
    print("\nSkipping Downstream Classification evaluation as test_loader is not available or test_dataset is empty.")


# 7. 保存最終分類器模型 (可選)
classifier_model_save_path = os.path.join(results_base_dir, "UNSW_NB15_downstream_classifier_final.pth")
torch.save(classifier_model.state_dict(), classifier_model_save_path)
print(f"\nFinal Downstream Classifier model saved to {classifier_model_save_path}")