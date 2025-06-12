from __future__ import print_function
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
#from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer, LabelEncoder
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import os
#print("Current working directory:", os.getcwd())
#print("File exists:", os.path.exists('kdd/binary/Training.csv'))

traindata = pd.read_csv('UNSWtrain.csv', header=0)
testdata = pd.read_csv('UNSWtest.csv', header=0)


# Drop irrelevant columns (optional: drop attack_cat if you're doing binary classification)
traindata = traindata.drop(columns=['id', 'attack_cat'])
testdata = testdata.drop(columns=['id', 'attack_cat'])

# Categorical columns that need encoding
categorical_cols = ['proto', 'service', 'state']

# Apply Label Encoding (fit on train+test, then transform both)
for col in categorical_cols:
    all_values = pd.concat([traindata[col], testdata[col]])
    encoder = LabelEncoder()
    encoder.fit(all_values)

    traindata[col] = encoder.transform(traindata[col])
    testdata[col] = encoder.transform(testdata[col])

# Now extract features and labels by name
X = traindata.drop(columns=['label'])
Y = traindata['label']

T = testdata.drop(columns=['label'])
C = testdata['label']

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

X_train = np.array(trainX)
y_train = np.array(Y)

X_test = np.array(testT)
y_test = np.array(C)

print("X_train shape:", X_train.shape)

'''

traindata = pd.read_csv('dnn1000/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('dnn1000/kdd/binary/Testing.csv', header=None)

X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)


X_train = np.array(trainX)
X_test = np.array(testT)
'''

batch_size = 64

model = Sequential()
model.add(Dense(1024, input_dim=42, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


'''
------------------------------------------AUTOENCODER----------------------------------
model = Sequential()
model.add(Dense(128, input_dim=42, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''
'''
-------------------------------------DEFAULT--------------------------------------------
# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn3layer/checkpoint-{epoch:02d}.keras", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn3layer/training_set_dnnanalysis.csv',separator=',', append=False)

# 建立必要的資料夾
os.makedirs("kddresults/dnn3layer", exist_ok=True)
os.makedirs("dnn/chung_results", exist_ok=True)

# 設定 callbacks
checkpointer = callbacks.ModelCheckpoint(
    filepath="kddresults/dnn3layer/checkpoint-{epoch:02d}.keras", 
    verbose=1, 
    save_best_only=True, 
    monitor='loss'
)
csv_logger = CSVLogger(
    'kddresults/dnn3layer/training_set_dnnanalysis.csv',
    separator=',',
    append=False
)
model.fit(X_train, y_train,  batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("dnn/chung_results/dnn3layer_model.hdf5")