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
source_path = os.getcwd() + '/Data/'
data_folder = 'Six_Actions_Updated_Train/'
data_folder_test = 'Six_Actions_Updated_Test/'

## ---   Fetch and split data into train and test   --- #
#csv_list_1 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*S.csv')]
#csv_list_2 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*C.csv')]
no_of_actions = 6
csv_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]
test_list = []
test_subjects = []
train_list = []

for i in range(0, 2):
    x = random.randint(1, int(len(csv_list)/no_of_actions) - 1)
    if x < 10:
        x = '0' + str(x) + '_'
        test_subjects.append(x)
    else:
        test_subjects.append(str(x) + '_')

for i in csv_list:
    #print("file:" +  i)
    train_flag = False
    for j in test_subjects:
        # print("if j: " + j + " in " + i)
        if j in i:
            # print('Yes')
            test_list.append(i)
            train_flag = True
    if train_flag == False:
        train_list.append(i)



# for i in range(0,2):
#     x = random.randint(0, int(len(csv_list)/no_of_actions) - 1)
#     test_subjects.append(x)

# for i in range(0,2):
#     x = random.randint(0, int(len(csv_list)/no_of_actions) - 1)
#     test_subjects.append(x)
#
# test_list = []
# train_list = []
# # Create training and testing lists of files
# # All actions included in the performance are held for testing
# for i in range(0,len(test_subjects)):
#     for f in os.listdir(source_path + data_folder):
#         if fnmatch.fnmatch(f, 'Set{}*.csv'.format(test_subjects[i])):
#             test_list.append(f)
#
# #train_list = csv_list.difference(*test_list)
# train_list = list(set(csv_list)^set(test_list))

##  ---  Define hyper parameters  ---  #
no_sensors = 4
features_per_sensor = 6
n_features = no_sensors * features_per_sensor
n_classes = 6  # number of outputs for classification
epochs = 25
learning_rate = 0.1

n_units = 100  # number of lstm cells

##  ---   Make motion windows   --- #
sliding_window_1 = 60
sliding_window_2 = [60, 80, 100, 120]
batch_size = 75


# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:  # range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-6].reset_index(drop=True).to_numpy()
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



## --- Training process  ---  #
print(' ------------- Training   ------------')
# for i in range(epochs):
#     #j = 0
#     avg_acc = 0
#     random.shuffle(train_list)
#     feature_list_main = list()
#     label_list_main = list()
feature_list_main = list()
label_list_main = list()
for file in train_list:
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
shuffle_list = list(zip(feature_list_main, label_list_main))
random.shuffle(shuffle_list)
feature_list_main, label_list_main = zip(*shuffle_list)
feature_list_main = np.reshape(feature_list_main,(len(feature_list_main),sliding_window_1,n_features))
label_list_main = np.array(label_list_main)
model.fit(feature_list_main, label_list_main,epochs=epochs, batch_size=n_batches, verbose=2)
model.summary()

model.save(source_path + 'Saved/Keras_05132020_2/', save_format='tf')
feature_list_main = []
label_list_main = []

## print('------  Validation  -------------')
for file in test_list:
    df = pd.read_csv(source_path + data_folder + file)
    # drop all Time columns
    df = df.drop(df.columns[[0, 4, 8, 12, 16, 20, 24, 28, 32]], axis=1)
    # create motion windows and separate data into input and output
    feature_df, label_df = create_motion_windows(sliding_window_1, df)
    feature_list_main.extend(feature_df)
    label_list_main.extend(label_df)

feature_list_main = np.reshape(feature_list_main,(len(feature_list_main),sliding_window_1,n_features))
label_list_main = np.array(label_list_main)
n_batches = int(total_motion_n_windows / batch_size)
#print('File: {}'.format(file))
model.evaluate(feature_list_main, label_list_main, batch_size=n_batches, verbose=2)

# ## Test script
# file = csv_list[0]
# # read the file into a data-frame
# df = pd.read_csv(source_path + data_folder + file)
# # drop all Time columns
# df = df.drop(df.columns[[0, 4, 8, 12, 16, 20, 24, 28, 32]], axis=1)
#
# # create motion windows and separate data into input and output
# # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)w
# feature_list, label_list = create_motion_windows(sliding_window_1, df)
# feature_list_nd = np.reshape(feature_list, (len(feature_list),sliding_window_1, n_features))
# print("Shape is: {}".format(feature_list_nd.shape))
#


# import tensorflow as tf
# from tensorflow.python.framework import convert_to_constants
# @tf.function(input_signature=[tf.TensorSpec(shape=[<input_shape>], dtype=tf.float32)])
# def to_save(x):
#     return model(x)
# f = to_save.get_concrete_function()
# constantGraph = convert_to_constants.convert_variables_to_constants_v2(f)
# tf.io.write_graph(constantGraph.graph.as_graph_def(), <output_dir>, <output_file>)