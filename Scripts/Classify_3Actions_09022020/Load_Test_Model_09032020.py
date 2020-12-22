## --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import time
import zmq
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model



# File path to the database files
source_path = os.getcwd() + '/../..' + '/Data/'
data_folder = '3 Actions_09022020/PreparedData/'
save_data_folder = '3 Actions_09022020/Saved/'



## ---   Fetch and split data into train and test   --- #
no_of_actions = 3
csv_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]
test_subjects = ['28','16','11']
test_list = []
# split data into test and train set
for i in test_subjects:
    for f in csv_list:
        if fnmatch.fnmatch(f, '*' + i + '.csv'):
            test_list.append(f)



## 1 ---  Define hyper parameters  ---  #
no_sensors = 1
features_per_sensor = 6
n_features = no_sensors * features_per_sensor
n_classes = 3 #number of outputs for classification
epochs = 12
learning_rate = 0.1

n_units = 100 # number of lstm cells

sliding_window = 60
batch_size = 120


# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-3].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step+no_windows, 6:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df


model = Sequential()
model.add(LSTM(n_units, input_shape=(sliding_window, n_features)))
# model.add(Dense(n_classes*2, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
loaded_model = load_model(source_path + save_data_folder, custom_objects=None, compile=True)

feature_list_main = []
label_list_main =[]
prediction_list = []

## print('------  Validation  -------------')
for file in test_list:
    df = pd.read_csv(source_path + data_folder + file)
    # drop all Time columns
    df = df.drop(df.columns[[0]], axis=1)
    # create motion windows and separate data into input and output
    feature_df, label_df = create_motion_windows(sliding_window, df)
    # feature_list_main.extend(feature_df)
    label_list_main.extend(label_df)

    feature_list_main = np.reshape(feature_df,(len(feature_df),sliding_window,n_features))
    # label_list_main = np.array(label_list_main)
    # n_batches = int(total_motion_n_windows / batch_size)
    #print('File: {}'.format(file))
    for win in feature_list_main:
        prediction = loaded_model.predict(feature_list_main)
        prediction_list.extend(prediction)

