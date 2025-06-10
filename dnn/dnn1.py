from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from tensorflow.keras.preprocessing import sequence
# from keras.utils import np_utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# KDDCUP99
# traindata = pd.read_csv('dnn/kdd/binary/Training.csv', header=None)
# testdata = pd.read_csv('dnn/kdd/binary/Testing.csv', header=None)
# X = traindata.iloc[:,1:42]
# Y = traindata.iloc[:,0]
# C = testdata.iloc[:,0]
# T = testdata.iloc[:,1:42]

# UNSW_NB15
traindata = pd.read_csv('./dataset/UNSW_NB15/Multiclass/UNSW_NB15_training_multiclass.csv')
testdata = pd.read_csv('./dataset/UNSW_NB15/Multiclass/UNSW_NB15_testing_multiclass.csv')

X = traindata.iloc[:,1:-1]
Y = traindata.iloc[:,-1]
C = testdata.iloc[:,-1]
T = testdata.iloc[:,1:-1]


trainX = np.array(X)
testT = np.array(T)

trainX.astype(float)
testT.astype(float)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)
testT = scaler.transform(testT)

y_train = np.array(Y)
y_test = np.array(C)

y_train = to_categorical(np.array(Y), num_classes=10)
y_test = to_categorical(np.array(C), num_classes=10)


X_train = np.array(trainX)
X_test = np.array(testT)


batch_size = 64

# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=42,activation='relu')) # kddcup99 has 41 features, # UNSW_NB15 has 42 features
model.add(Dropout(0.01))
model.add(Dense(10))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# for multiclass classification
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="dnn/kddresults/dnn1layer/checkpoint-{epoch:02d}.keras", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('dnn/kddresults/dnn1layer/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("dnn/kddresults/dnn1layer/dnn1layer_model.keras")








