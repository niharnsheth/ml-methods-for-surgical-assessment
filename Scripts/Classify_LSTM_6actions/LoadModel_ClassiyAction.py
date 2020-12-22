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
from tensorflow.keras.models import load_model

# File path to the database files
source_path = os.getcwd() + '/Data/'
data_folder = '4sensors_6actions_dataset/Prepared Data/'
saved_model_folder = 'Saved/TF2_07092020/'
file_path = source_path + saved_model_folder
## ---   Fetch and split data into train and test   --- #
no_of_actions = 3
csv_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]

##  ---  Define hyper parameters  ---  #
no_sensors = 4
features_per_sensor = 6
n_features = no_sensors * features_per_sensor
n_classes = 3 #number of outputs for classification
epochs = 10
learning_rate = 0.1

n_units = 100 # number of lstm cells

##  ---   Make motion windows   --- #
sliding_window_1 = 60
sliding_window_2 = [60, 80, 100, 120]
batch_size = 25


# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-3].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step+no_windows, 24:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df


## --- tf.Keras implementation of LSTM layers --- #
model = Sequential()
model.add(LSTM(n_units, input_shape=(sliding_window_1, n_features)))
# model.add(Dense(n_classes*2, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

loaded_model = load_model(file_path, custom_objects=None, compile=True)

feature_list_main = list()
label_list_main = list()

file = csv_list[0]
# read the file into a data-frame
df = pd.read_csv(source_path + data_folder + file)
# drop all Time columns
df = df.drop(df.columns[[0, 4, 8, 12, 16, 20, 24, 28, 32]], axis=1)

# create motion windows and separate data into input and output
# feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)w
feature_list, label_list = create_motion_windows(sliding_window_1, df)

feature_list_main.extend(feature_list)
label_list_main.extend(label_list)

total_motion_n_windows = len(feature_list_main)
n_batches = int(total_motion_n_windows / batch_size)

feature_list_main = np.reshape(feature_list_main,(len(feature_list_main),sliding_window_1,n_features))
predictions = loaded_model.predict(feature_list_main)

print(predictions[0])
