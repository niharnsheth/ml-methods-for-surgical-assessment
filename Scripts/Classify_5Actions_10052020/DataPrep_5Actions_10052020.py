# Script prepares data for training. Missing columns for sensors are added.

import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
from sqlite3 import Error

# File path to the database files
source_path = os.getcwd() + '/../..' + '/Data/'
save_to_folder = '5 Actions_10032020/PreparedData/'
data_folder_list = ['5 Actions_10032020/OriginalData/ChloraPrep/',
                    '5 Actions_10032020/OriginalData/NeedleInsertion/',
                    '5 Actions_10032020/OriginalData/Dilator/',
                    '5 Actions_10032020/OriginalData/Cut/',
                    '5 Actions_10032020/OriginalData/Tracing/']

data_folder_example = '3 Actions_09022020/Dilator/'

# csv_list_1 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]
# file = csv_list_1[0]
no_of_actions = 5



##  ---   Normalize the data  --- #
# min and max values for each feature
#pos_x = (-9, 11)  # ideal +-28
pos_x = (-16,11)
pos_y = (-8, 20)  # ideal +-30
pos_z = (-42, -15)  # ideal +20 +60


#  input msx and min to normalize data
def normalize_column_custom(min, max, input_df, feature_list):
    df_copy = input_df.copy()
    for feature in feature_list:
        df_copy[feature] = df_copy[feature].map(lambda a: normalize_input(a, min, max))
    return df_copy


# def normalize_dataframe_custom((maxX,minX),(maxY,minY),(maxZ,minZ),(maxA,minA),(maxB,minB),(maxG,minG),input_df):
def normalize_input(x, x_min, x_max):
    if x >= x_max:
        return 1
    elif x <= x_min:
        return 0
    else:
        return (x - x_min) / (x_max - x_min)


def scale_input(value, scale_value):
    if value > 0:
        return scale_value - value
    else:
        return scale_value + value


# use max and min values in sequence to normalize data
def normalize_column(input_df, feature_name):
    df_copy = input_df.copy()
    for feature in feature_name:
        max_value = input_df[feature].max()
        min_value = input_df[feature].min()
        if max_value == min_value:
            print("Error: Cannot normalize when max and min values are equal")
            return df_copy
        df_copy[feature] = (input_df[feature] - min_value) / (max_value - min_value)
    return df_copy


## Convert euler to quaternions
def euler_to_quaternion(roll, pitch, yaw):
    c1 = np.cos(roll/2)
    c2 = np.cos(pitch/2)
    c3 = np.cos(yaw/2)
    s1 = np.sin(roll/2)
    s2 = np.sin(pitch/2)
    s3 = np.sin(yaw/2)
    quat_w = c1*c2*c3 - s1*s2*s3
    quat_x = s1*s2*c3 + c1*c2*s3
    quat_y = s1*c2*c3 + c1*s2*s3
    quat_z = c1*s2*c3 - s1*c2*s3
    return [quat_x, quat_y, quat_z, quat_w]


## Create labels
def create_label_df(my_df, number_of_actions, action_index):
    # Create one-hot vector labels df
    num_of_rows = len(my_df.index)
    labels_arr = np.zeros((num_of_rows, number_of_actions))
    labels_arr[:, action_index] = 1
    return labels_arr


## Clean and Normalize data
for i in range(len(data_folder_list)):
    action_no = i
    csv_list_1 = [f for f in os.listdir(source_path + data_folder_list[i]) if fnmatch.fnmatch(f, '*.csv')]
    for file in csv_list_1:
        print("File updating: {}".format(file))
        # read data from csv into a dataframe
        df = pd.read_csv(source_path + data_folder_list[i] + file)
        df = df.drop(columns=['SId','PT','OT'], axis=1)

        # normalize position and orientation values
        df = normalize_column_custom(pos_x[0], pos_x[1], df, ['X'])
        df = normalize_column_custom(pos_y[0], pos_y[1], df, ['Y'])
        df = normalize_column_custom(pos_z[0], pos_z[1], df, ['Z'])
        #df = normalize_column_custom(ang_x[0], ang_x[1], df, ['A'])
        #df = normalize_column_custom(ang_y[0], ang_y[1], df, ['B'])
        #df_test = scale_column_custom(ang_z_scale, df, ['G'])
        #df = normalize_column_custom(ang_z[0], ang_z[1], df, ['G'])
        euler_arr = df[['A','B','G']].to_numpy()
        quat_arr = np.empty([len(df.index),4])
        for row in range(len(df.index)):
            quat_arr[row,:] = euler_to_quaternion(euler_arr[row,0],
                                                  euler_arr[row,1],
                                                  euler_arr[row,2])

        df_2 = pd.DataFrame(quat_arr)
        df = df.drop(columns=['A','B','G'], axis=1)
        main_df = np.hstack((df,df_2))

        label_arr = create_label_df(df, no_of_actions, action_no)
        main_df = np.hstack((main_df, label_arr))

        header_list = ['X1', 'Y1', 'Z1', 'PT1', 'A1', 'B1', 'G1', 'OT1',
                       'X2', 'Y2', 'Z2', 'PT2', 'A2', 'B2', 'G2', 'OT2']
        final_df = pd.DataFrame(main_df)
        final_df.to_csv(source_path + save_to_folder + file)
