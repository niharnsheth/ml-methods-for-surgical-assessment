## --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import csv
from tensorflow.python import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# File path to the database files
source_path = os.getcwd() + '/../..' + '/Data/'
data_folder = '4sensors_6actions_dataset/Prepared Data/'
#data_folder_test = 'Six_Actions_Updated_Test/'


## ---   Fetch and split data into train and test   --- #
no_of_actions = 6
csv_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]

test_list = []
test_subjects = []
train_list = []
# randomly select subject for testing network
for i in range(0, 2):
    x = random.randint(1, int(len(csv_list)/no_of_actions) - 1)
    if x < 10:
        x = '0' + str(x)
        test_subjects.append(x)
    else:
        test_subjects.append(str(x))
# split data into test and train set
for i in test_subjects:
    for f in csv_list:
        if fnmatch.fnmatch(f, '*' + i + '.csv'):
            test_list.append(f)
        else:
            train_list.append(f)


## 1 ---  Define hyper parameters  ---  #
no_sensors = 2
features_per_sensor = 6
n_features = no_sensors * features_per_sensor
n_classes = 6 #number of outputs for classification
epochs = 12
learning_rate = 0.1

n_units = 100 # number of lstm cells

sliding_window_1 = 60
sliding_window_2 = [60, 80, 100, 120]
batch_size = 120
##  ---   Make motion windows   --- #

# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-6].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step+no_windows, 12:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df


## --- tf.Keras implementation of LSTM layers --- #
model = Sequential()
model.add(LSTM(n_units, input_shape=(sliding_window_1, n_features)))
# model.add(Dense(n_classes*2, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

## --- Training process  ---  #
print(' ------------- Training   ------------')

feature_list_main = list()
label_list_main = list()
for file in train_list:
    # read the file into a data-frame
    df = pd.read_csv(source_path + data_folder + file)
    # drop all Time columns
    df = df.drop(df.columns[[0, 4, 8, 12,16]], axis=1)

    # create motion windows and separate data into input and output
    # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)w
    feature_list, label_list = create_motion_windows(sliding_window_1, df)

    feature_list_main.extend(feature_list)
    label_list_main.extend(label_list)
total_motion_n_windows = len(feature_list_main)
n_batches = int(total_motion_n_windows / batch_size)
shuffle_list = list(zip(feature_list_main, label_list_main))
random.shuffle(shuffle_list)
feature_list_main, label_list_main = zip(*shuffle_list)
feature_list_main = np.reshape(feature_list_main,(len(feature_list_main),sliding_window_1,n_features))
label_list_main = np.array(label_list_main)
model.fit(feature_list_main, label_list_main,epochs=epochs, batch_size=n_batches, verbose=2)
model.summary()

model.save(source_path + 'Saved/TF2_08122020_2/', save_format='tf')
feature_list_main = []
label_list_main = []

## print('------  Validation  -------------')
for file in test_list:
    df = pd.read_csv(source_path + data_folder + file)
    # drop all Time columns
    df = df.drop(df.columns[[0, 4, 8, 12, 16]], axis=1)
    # create motion windows and separate data into input and output
    feature_df, label_df = create_motion_windows(sliding_window_1, df)
    feature_list_main.extend(feature_df)
    label_list_main.extend(label_df)

feature_list_main = np.reshape(feature_list_main,(len(feature_list_main),sliding_window_1,n_features))
label_list_main = np.array(label_list_main)
n_batches = int(total_motion_n_windows / batch_size)
#print('File: {}'.format(file))
model.evaluate(feature_list_main, label_list_main, batch_size=n_batches, verbose=2)
