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

# File path to the database files
source_path = os.getcwd() + '/Data/'
data_folder = 'Six_Actions_Updated_Train/'
data_folder_test = 'Six_Actions_Updated_Test/'



## ---   Fetch and split data into train and test   --- #
no_of_actions = 6
test_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]

##  ---  Define hyper parameters  ---  #
no_sensors = 4
features_per_sensor = 6
n_features = no_sensors * features_per_sensor
n_classes = 6 #number of outputs for classification

n_units = 150 # number of lstm cells

##  ---   Make motion windows   --- #
sliding_window_1 = 80
sliding_window_2 = [60, 80, 100, 120]
batch_size = 50


# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-6].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step+no_windows, 24:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df


sess = tf.Session()
# Load meta graph and get wieghts
saver = tf.train.import_meta_graph(source_path + 'Saved/Trial_1/my-test-model.meta')
saver.restore(sess,tf.train.latest_checkpoint(source_path + 'Saved/Trial_1'))

graph = tf.get_default_graph()
weight = graph.get_tensor_by_name('w1')
bias = graph.get_tensor_by_name('b1')


##  ---  Create placeholders for features and labels  --- #
#input_data = tf.placeholder(tf.float32, [None, None, n_features])
input_data = tf.placeholder(tf.float32, [None, sliding_window_1, n_features])
output_data = tf.placeholder(tf.float32, [None, n_classes])
