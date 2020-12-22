# Script prepares data for traning. Missing columns for sensors are added.

import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
from sqlite3  import Error

# File path to the database files
source_path = os.getcwd() + '/Data/'
data_folder = '4sensors_6actions_dataset/Suturing_action6_07082020/'
save_to_folder = '4sensors_6actions_dataset/Prepared Data/'

csv_list_1 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]
#file = csv_list_1[0]
no_of_actions = 6
action_no = 5
def CreateLabelDf(number_of_actions, action_index):
    # Create one-hot vector labels df
    labels_df = np.zeros((longest_data_sequence_length, number_of_actions))
    labels_df[:, action_index] = 1
    return labels_df

##  ---   Normalize the data  --- #
# min and max values for each feature
pos_x = (10,-10)  # ideal +-28
pos_y = (10,-5)  # ideal +-30
pos_z = (-5,-25)  # ideal +20 +60
ang_x = (90,-90)
ang_y = (180,-180)
ang_z = (180,-180)

#  input msx and min to normalize data
def normalize_column_custom(max, min, input_df, feature_list):
    df_copy = input_df.copy()
    for feature in feature_list:
        df_copy[feature] = df_copy[feature].map(lambda a: scale_input(a, max, min))
        # df_copy[feature] = (input_df[feature] - min) / (max - min)
    return df_copy

# def normalize_dataframe_custom((maxX,minX),(maxY,minY),(maxZ,minZ),(maxA,minA),(maxB,minB),(maxG,minG),input_df):

def scale_input(x, max, min):
    if x >= max:
        return 1
    elif x <= min:
        return 0
    else:
        return (x - min)/(max - min)

# use max and min values in sequence to normalize data
def normalize_column(input_df, feature_name):
    df_copy = input_df.copy()
    for feature in feature_name:
        max_value = df[feature].max()
        min_value = df[feature].min()
        if max_value == min_value:
            print("Error: Cannot normalize when max and min values are equal")
            return df_copy
        df_copy[feature] = (df[feature] - min_value) / (max_value - min_value)
    return df_copy

def factorial_pos_no(number):
    factorial = 1
    for i in range(1, number + 1):
        factorial = factorial * i
    return factorial

## Clean and Normalize data
for file in csv_list_1:
    print("File updating: {}".format(file))
    # read data from csv into a dataframe
    df = pd.read_csv(source_path + data_folder + file)
    # separate position and orientation data
    df_pos = df.iloc[:, [0, 1, 2, 3, 4]]
    df_pos = df_pos.rename(columns={0: 'SId', 1: 'X', 2: 'Y', 3: 'Z', 4: 'Pt'})
    df_ori = df.iloc[:, [0, 5, 6, 7, 8]]
    df_ori = df_ori.rename(columns={0: 'SId', 5: 'A', 6: 'B', 7: 'G', 8: 'Ot'})
    # Find Number of sensors used in task
    sensors_ids = df['SId'].unique()
    sensors_ids = np.sort(sensors_ids)
    list_length = len(sensors_ids)
    list_pos_df = [0 for i in range(len(sensors_ids))]
    list_ori_df = [0 for i in range(len(sensors_ids))]
    list_pos_numpy = [0 for i in range(len(sensors_ids))]
    list_ori_numpy = [0 for i in range(len(sensors_ids))]
    print("Sensors used: {}".format(sensors_ids))

    # Traverse through each dataframe to clean data
    for i in range(list_length):
        # Separate data based on SensorId before padding
        list_pos_df[i] = df_pos.loc[df_pos['SId'] == sensors_ids[i]]
        list_ori_df[i] = df_ori.loc[df_ori['SId'] == sensors_ids[i]]
        # Remove all Nan value rows
        list_pos_df[i] = list_pos_df[i].dropna(thresh=3)
        list_ori_df[i] = list_ori_df[i].dropna(thresh=3)
        # normalize position and orientation values
        list_pos_df[i] = normalize_column_custom(pos_x[0], pos_x[1], list_pos_df[i], ['X'])
        list_pos_df[i] = normalize_column_custom(pos_y[0], pos_y[1], list_pos_df[i], ['Y'])
        list_pos_df[i] = normalize_column_custom(pos_z[0], pos_z[1], list_pos_df[i], ['Z'])
        list_ori_df[i] = normalize_column_custom(ang_x[0], ang_x[1], list_ori_df[i], ['A'])
        list_ori_df[i] = normalize_column_custom(ang_y[0], ang_y[1], list_ori_df[i], ['B'])
        list_ori_df[i] = normalize_column_custom(ang_z[0], ang_z[1], list_ori_df[i], ['G'])

        # Convert all to numpy_arrays for padding
        list_pos_numpy[i] = list_pos_df[i].to_numpy()
        list_ori_numpy[i] = list_ori_df[i].to_numpy()

    # Padding overall sensor values to have uniform length of data
    longest_data_sequence_length = 0
    longest_data_sequence = 0
    # find the longest data length
    for i in range(list_length):
        if list_pos_numpy[i].shape[0] >= longest_data_sequence_length:
            longest_data_sequence_length = list_pos_numpy[i].shape[0]
            longest_data_sequence = i
            print("Longest length = {}".format(longest_data_sequence_length))
            print("Longest index = {}".format(longest_data_sequence))
    # pad values for all shorter arrays
    for i in range(list_length):
        if i == longest_data_sequence:
            print("No need to pad list index: {}".format(i))
            #break
        else:
            no_pad_rows = longest_data_sequence_length - list_pos_numpy[i].shape[0]
            list_pos_numpy[i] = np.pad(list_pos_numpy[i],((0,no_pad_rows),(0,0)),'edge')
            list_ori_numpy[i] = np.pad(list_ori_numpy[i], ((0, no_pad_rows), (0, 0)), 'edge')
            print("Rows to pad: {}".format(no_pad_rows))
            print("Final length after padding: {}".format(list_pos_numpy[i].shape[0]))
    # remove the SId column
    for i in range(list_length):
        list_pos_numpy[i] = np.delete(list_pos_numpy[i], 0, axis=1)
        list_ori_numpy[i] = np.delete(list_ori_numpy[i], 0, axis=1)


    # Create final dataset and save to csv
    total_no_sensors = 4
    features_per_sensor = 8
    total_features = total_no_sensors * features_per_sensor
    zero_value_numpy = 4 - list_length
    list_zero_numpy = [0 for i in range(zero_value_numpy)]

    # Calculate permutations:
    poss_permutations = factorial_pos_no(total_no_sensors)/factorial_pos_no(total_no_sensors - list_length)
   
    for i in range(zero_value_numpy):
        list_zero_numpy[i] = np.zeros((longest_data_sequence_length,features_per_sensor))

    append_sequence = [0, 1, 2, 3]
    #final_numpy = np.empty((longest_data_sequence_index,total_features))
    final_numpy = np.zeros((longest_data_sequence_length,1))
    #sensor_sequence_index = 0
    sequence_index = 0
    sensor_index = 0

    # add zero value to 
    for i in append_sequence:
        if i == sensors_ids[sensor_index]:
            print("i = {} j = {}".format(i, sensor_index))
            final_numpy = np.hstack((final_numpy, list_pos_numpy[sensor_index]))
            final_numpy = np.hstack((final_numpy, list_ori_numpy[sensor_index]))
            if sensor_index < len(sensors_ids) - 1:
                sensor_index += 1
        else:
            print("i = {} k = {}".format(i, sequence_index))
            final_numpy = np.hstack((final_numpy, list_zero_numpy[sequence_index]))
            if sequence_index < len(list_zero_numpy) - 1:
                sequence_index += 1
    final_numpy = np.delete(final_numpy,0,axis=1)

    label_df = CreateLabelDf(no_of_actions, action_no)
    final_numpy = np.hstack((final_numpy,label_df))
    header_list = ['X1','Y1','Z1','PT1', 'A1','B1','G1','OT1',
                   'X2','Y2','Z2','PT2', 'A2','B2','G2','OT2',
                   'X3','Y3','Z3','PT3', 'A3','B3','G3','OT3',
                   'X4','Y4','Z4','PT4', 'A4','B4','G4','OT4']
    final_df = pd.DataFrame(final_numpy)
    final_df.to_csv(source_path + save_to_folder + file)
