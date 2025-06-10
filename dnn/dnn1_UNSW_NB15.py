from __future__ import print_function
# from sklearn.model_selection import train_test_split # 目前未使用，可移除
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
# from tensorflow.keras.preprocessing import sequence # 目前未使用，可移除
from tensorflow.keras.utils import to_categorical # Keras 2.9+ 建議用 tensorflow.keras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation # Embedding, LSTM, SimpleRNN, GRU 未使用，可移除
# from keras.datasets import imdb # 目前未使用，可移除
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, confusion_matrix) # 移除了 MSE, MAE 因為是分類問題
# from sklearn import metrics # precision_score 等已從 sklearn.metrics 導入
from sklearn.preprocessing import StandardScaler # 移除了 Normalizer 因為未使用
# import h5py # 目前未使用，可移除
from tensorflow.keras import callbacks # Keras 2.9+ 建議用 tensorflow.keras.callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger # EarlyStopping, ReduceLROnPlateau 未使用，可移除
import os # 用於創建目錄

# --- 數據集選擇 ---
# 設置一個標誌來選擇數據集，方便切換
DATASET_CHOICE = "UNSW_NB15" # 或 "KDDCUP99"

if DATASET_CHOICE == "KDDCUP99":
    print("Using KDDCUP99 dataset")
    traindata = pd.read_csv('dnn/kdd/binary/Training.csv', header=None) # 請確保路徑正確
    testdata = pd.read_csv('dnn/kdd/binary/Testing.csv', header=None)   # 請確保路徑正確
    # 假設 KDDCUP99 的標籤是第一列，特徵是從第二列開始
    X_train_raw = traindata.iloc[:, 1:]
    Y_train_raw = traindata.iloc[:, 0]
    X_test_raw = testdata.iloc[:, 1:]
    C_test_raw = testdata.iloc[:, 0]
    input_dim_dynamic = X_train_raw.shape[1] # 動態獲取特徵維度
    num_classes_dynamic = len(np.unique(Y_train_raw)) # 動態獲取類別數 (假設是二分類)
    # KDDCUP99 的標籤可能是字串，需要轉換
    # 例如，如果標籤是 'normal.' 和 'attack.'
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # Y_train_raw = le.fit_transform(Y_train_raw)
    # C_test_raw = le.transform(C_test_raw)

elif DATASET_CHOICE == "UNSW_NB15":
    print("Using UNSW_NB15 dataset")
    traindata = pd.read_csv('./dataset/UNSW_NB15/Multiclass/UNSW_NB15_training_multiclass.csv')
    testdata = pd.read_csv('./dataset/UNSW_NB15/Multiclass/UNSW_NB15_testing_multiclass.csv')
    # 假設特徵是從第二列到倒數第二列，最後一列是標籤
    X_train_raw = traindata.iloc[:, 1:-1]
    Y_train_raw = traindata.iloc[:, -1]
    X_test_raw = testdata.iloc[:, 1:-1]
    C_test_raw = testdata.iloc[:, -1]
    input_dim_dynamic = X_train_raw.shape[1] # 動態獲取特徵維度: 42
    num_classes_dynamic = 10 # UNSW_NB15 多分類通常是10類
else:
    raise ValueError("Invalid DATASET_CHOICE. Choose 'KDDCUP99' or 'UNSW_NB15'.")

print(f"Dataset: {DATASET_CHOICE}")
print(f"Input dimension: {input_dim_dynamic}")
print(f"Number of classes: {num_classes_dynamic}")
print(f"Shape of X_train_raw: {X_train_raw.shape}")
print(f"Shape of Y_train_raw: {Y_train_raw.shape}")
print(f"Shape of X_test_raw: {X_test_raw.shape}")
print(f"Shape of C_test_raw: {C_test_raw.shape}")

# --- 數據預處理 ---
# 轉換為 NumPy 數組並確保是浮點型
trainX_np = np.array(X_train_raw).astype(float)
testT_np = np.array(X_test_raw).astype(float)

# 標準化 (StandardScaler)
# 1. Fit a scaler on the TRAINING data only
scaler = StandardScaler().fit(trainX_np)
# 2. Apply the scaler to the TRAINING data
trainX_scaled = scaler.transform(trainX_np)
# 3. Apply the SAME scaler to the TESTING data
testT_scaled = scaler.transform(testT_np)

# 處理標籤
print(f"Unique labels in Y_train_raw: {np.unique(Y_train_raw)}")
print(f"Unique labels in C_test_raw: {np.unique(C_test_raw)}")

y_train_cat = to_categorical(np.array(Y_train_raw), num_classes=num_classes_dynamic)
y_test_cat = to_categorical(np.array(C_test_raw), num_classes=num_classes_dynamic)

# 最終的訓練和測試集
X_train_final = np.array(trainX_scaled)
X_test_final = np.array(testT_scaled)

print(f"Shape of X_train_final: {X_train_final.shape}")
print(f"Shape of y_train_cat: {y_train_cat.shape}")
print(f"Shape of X_test_final: {X_test_final.shape}")
print(f"Shape of y_test_cat: {y_test_cat.shape}")

# --- 模型定義與訓練 ---
batch_size = 64
epochs = 100 # 你原本設定為 100

# 創建結果目錄
results_dir = "dnn/kddresults/dnn1layer"
os.makedirs(results_dir, exist_ok=True)

# 1. 定義網路
model = Sequential()
model.add(Dense(1024, input_dim=input_dim_dynamic, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(num_classes_dynamic)) # 輸出層的神經元數量等於類別數
model.add(Activation('softmax')) # 多分類問題通常使用 softmax 激活函數

# 編譯模型
# 對於多類別 one-hot 編碼的標籤，使用 'categorical_crossentropy'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary() # 打印模型結構

# 回調函數
checkpoint_path = os.path.join(results_dir, "checkpoint-{epoch:02d}.keras")
csv_log_path = os.path.join(results_dir, "training_set_dnnanalysis.csv")

checkpointer = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=True,
    monitor='val_loss' # 通常監控驗證集損失以保存最佳模型
)
csv_logger = CSVLogger(csv_log_path, separator=',', append=False)

# 訓練模型
# 加入 validation_data 以便 ModelCheckpoint 可以監控 val_loss
# 如果你沒有單獨的驗證集，可以從訓練集中分割一部分，或者直接使用測試集作為驗證 (不推薦用於最終評估，但可用於開發)
# 為了簡單起見，這裡直接使用測試集作為驗證數據，但要注意這會讓 val_loss/val_accuracy 反映測試集性能
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train_final,
    y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpointer, csv_logger],
    validation_data=(X_test_final, y_test_cat) # 添加驗證數據
)

# 保存最終模型
final_model_path = os.path.join(results_dir, "dnn1layer_model_final.keras")
model.save(final_model_path)
print(f"\nFinal model saved to {final_model_path}")

# --- 模型測試 (Evaluation) ---
print("\n--- Starting Model Evaluation on Test Set ---")

# 1. 使用 Keras 的 evaluate 方法獲取損失和準確率
loss, acc = model.evaluate(X_test_final, y_test_cat, batch_size=batch_size, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy (from model.evaluate): {acc:.4f}")

# 2. 進行預測以計算更詳細的指標
y_pred_proba = model.predict(X_test_final, batch_size=batch_size)
y_pred_classes = np.argmax(y_pred_proba, axis=1) # 將機率轉換為類別索引

# 真實標籤也需要是類別索引 (而不是 one-hot)
y_true_classes = np.argmax(y_test_cat, axis=1)

# 計算 sklearn 指標
test_accuracy_sklearn = accuracy_score(y_true_classes, y_pred_classes)
# average='weighted' 考慮了類別不平衡
test_precision_weighted = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
test_recall_weighted = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
test_f1_weighted = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
# average='macro' 不考慮類別不平衡，對所有類別同等對待
test_f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

print(f"Test Accuracy (from sklearn): {test_accuracy_sklearn:.4f}")
print(f"Test Precision (weighted): {test_precision_weighted:.4f}")
print(f"Test Recall (weighted): {test_recall_weighted:.4f}")
print(f"Test F1-score (weighted): {test_f1_weighted:.4f}")
print(f"Test F1-score (macro): {test_f1_macro:.4f}")

# 混淆矩陣
print("\nConfusion Matrix (rows: True Labels, cols: Predicted Labels):")
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

# 可以將結果保存到文件
results_summary_path = os.path.join(results_dir, "test_results_summary.txt")
with open(results_summary_path, "w") as f:
    f.write(f"Dataset: {DATASET_CHOICE}\n")
    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy (from model.evaluate): {acc:.4f}\n")
    f.write(f"Test Accuracy (from sklearn): {test_accuracy_sklearn:.4f}\n")
    f.write(f"Test Precision (weighted): {test_precision_weighted:.4f}\n")
    f.write(f"Test Recall (weighted): {test_recall_weighted:.4f}\n")
    f.write(f"Test F1-score (weighted): {test_f1_weighted:.4f}\n")
    f.write(f"Test F1-score (macro): {test_f1_macro:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm, separator=', '))
print(f"\nTest results summary saved to {results_summary_path}")

print("\n--- Script Finished ---")