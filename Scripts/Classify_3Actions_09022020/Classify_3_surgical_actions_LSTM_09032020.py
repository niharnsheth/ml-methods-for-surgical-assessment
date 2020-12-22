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
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

# File path to the database files
source_path = os.getcwd() + '/../..' + '/Data/'
data_folder = '3 Actions_09022020/PreparedData/'
save_data_folder = '3 Actions_09022020/Saved/'
#data_folder_test = 'Six_Actions_Updated_Test/'


## ---   Fetch and split data into train and test   --- #
no_of_actions = 3
csv_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]

test_list = []
test_subjects = []
train_list = []
# randomly select subject for testing network
for i in range(0, 3):
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
no_sensors = 1
features_per_sensor = 7
n_features = no_sensors * features_per_sensor
n_classes = 3 #number of outputs for classification
epochs = 5
learning_rate = 0.001

n_units = 100 # number of lstm cells

sliding_window = 60
batch_size = 100

##  ---   Make motion windows   --- #

# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-3].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step+no_windows, 7:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df


## --- tf.Keras implementation of LSTM layers --- #
model = Sequential()
opt = tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs)
model.add(LSTM(n_units, input_shape=(sliding_window, n_features), name='lstm_layer'))
model.add(Dropout(0.5))
model.add(Dense(n_units, activation='relu', name='projeciton_layer'))
model.add(Dense(n_classes, activation='softmax', name='output_layer'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## --- Training process  ---  #
print(' ------------- Training   ------------')

feature_list_main = list()
label_list_main = list()
for file in train_list:
    # read the file into a data-frame
    df = pd.read_csv(source_path + data_folder + file)
    # drop index columns
    df = df.drop(df.columns[[0]], axis=1)

    # create motion windows and separate data into input and output
    # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)w
    feature_list, label_list = create_motion_windows(sliding_window, df)
    feature_list_main.extend(feature_list)
    label_list_main.extend(label_list)

total_motion_n_windows = len(feature_list_main)
n_batches = int(total_motion_n_windows / batch_size)
shuffle_list = list(zip(feature_list_main, label_list_main))
random.shuffle(shuffle_list)
feature_list_main, label_list_main = zip(*shuffle_list)
feature_list_main = np.reshape(feature_list_main,(len(feature_list_main),sliding_window,n_features))
label_list_main = np.array(label_list_main)
model.fit(feature_list_main, label_list_main, epochs=epochs, batch_size=batch_size, verbose=2)
model.summary()

model.save(source_path + save_data_folder, save_format='tf')
#feature_list_main = []
#label_list_main = []

## print('------  Validation  -------------')
for file in test_list:
    df = pd.read_csv(source_path + data_folder + file)
    # drop all Time columns
    df = df.drop(df.columns[[0]], axis=1)
    # create motion windows and separate data into input and output
    feature_input_list, label_out_list = create_motion_windows(sliding_window, df)

    #feature_list_main.extend(feature_df)
    #label_list_main.extend(label_df)
    #feature_list_main = np.reshape(feature_list_main,(len(feature_list_main),sliding_window,n_features))
    #label_list_main = np.array(label_list_main)
    feature_input_list = np.reshape(feature_input_list,(len(feature_input_list),sliding_window,n_features))
    label_out_list = np.array(label_out_list)
    n_batches = int(total_motion_n_windows / batch_size)
    print('Evaluationg File: {}'.format(file))

    #model.evaluate(feature_list_main, label_list_main, batch_size=n_batches, verbose=2)
    model.evaluate(feature_input_list, label_out_list, batch_size=n_batches, verbose=2)
