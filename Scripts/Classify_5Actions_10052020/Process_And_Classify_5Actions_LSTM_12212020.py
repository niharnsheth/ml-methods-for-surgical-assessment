# ==================================================================================================================== #
# ========================= CLASSIFYING SURGICAL ACTIONS FROM MOTION DATA ===========================================  #


#%% =========================================== Importing Libraries ================================================ %%#
# -------------------------------------- Import all necessary libraries -----------------------------------------------#

# File imports and aggregates data from multiple databases
import os
import fnmatch


import pandas as pd
import numpy as np
import quaternion as qt


import tensorflow as tf
import random
from tensorflow.python import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#%% ============================================== Import Data ===================================================== %%#
# --------------------------------------- Import data from separate folders -------------------------------------------#

# File path to the database files

#source_path = os.getcwd() + '/../..' + '/Data/'
source_path = os.getcwd()  + '/Data/'
#save_to_folder = '5 Actions_10032020/PreparedData/'
data_folder_list = ['5 Actions_10032020/OriginalData/ChloraPrep/',
                    '5 Actions_10032020/OriginalData/NeedleInsertion/',
                    '5 Actions_10032020/OriginalData/Dilator/',
                    '5 Actions_10032020/OriginalData/Cut/',
                    '5 Actions_10032020/OriginalData/Tracing/']
data_folder_example = '3 Actions_09022020/Dilator/'
save_data_folder = '5 Actions_10032020/Saved/12152020_1/'
# Data array
df = []
labels = []

for i in range(len(data_folder_list)):
    # get list of samples for action
    #action_no = i
    csv_list_1 = [f for f in os.listdir(source_path + data_folder_list[i]) if fnmatch.fnmatch(f, '*.csv')]
    # read and import csv files
    for file in csv_list_1:
        print("File updating: {}".format(file))
        # read data from csv into a dataframe
        file_df = pd.read_csv(source_path + data_folder_list[i] + file)
        df.append(file_df.drop(columns=['SId'], axis=1))
        labels.append(i)


#%% =================================== Extract input features and output labels =================================== %%#
# ------------------------ Add features: linear and angular velocities to input data ----------------------------------#
# -------------------------------------------- Create one-hot vector labels ------------------------------------------ #

# Convert euler to quaternions
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
    return [quat_w, quat_x, quat_y, quat_z]


# Calculates linear and angular velocity for given values
def calculate_velocity(x1,x2,t1,t2):
    return (x2 - x1) / (t2 - t1)


# Calculate velocity for a list of consecutive values
def cal_vel_for_range(pos_ori_values, time_values):
    # Check length of lists
    if len(pos_ori_values) == len(time_values):
        # initialize local variables
        velocity = [0]
        val = 0
        # calculate velocity and add to list
        while val < len(pos_ori_values) - 1:
            # print(pos_ori_values[val])
            # print(pos_ori_values[val+1])
            velocity.append(calculate_velocity(pos_ori_values[val],
                                               pos_ori_values[val+1],
                                               time_values[val],
                                               time_values[val+1]))
            val += 1
        return velocity


for file_no in range(len(df)):
    # calculate absolute linear velocities
    pos_values_array = df[file_no][['X', 'Y', 'Z', 'PT']].to_numpy()
    x_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 0], pos_values_array[:,-1]))
    y_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 1], pos_values_array[:,-1]))
    z_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 2], pos_values_array[:,-1]))

    # convert euler angles to quaternion
    euler_angles_arr = df[file_no][['A','B','G','OT']].to_numpy()
    np_quaternions_arr = np.empty([len(df[file_no].index),4])
    for row in range(len(df[file_no].index)):
        np_quaternions_arr[row,:] = euler_to_quaternion(euler_angles_arr[row,0],
                                                        euler_angles_arr[row,1],
                                                        euler_angles_arr[row,2])

    # calculate the angular velocities
    quat_arr = qt.as_quat_array(np_quaternions_arr)
    ang_velocity_quat_arr = qt.quaternion_time_series.angular_velocity(quat_arr, euler_angles_arr[:,-1])
    ang_velocity_quat_arr = np.absolute(ang_velocity_quat_arr)

    partial_features = np.hstack((pos_values_array[:,0:3],np_quaternions_arr))
    partial_features = np.hstack((partial_features,np.c_[x_linear_velocity],
                                  np.c_[y_linear_velocity],
                                  np.c_[z_linear_velocity]))
    partial_features = np.hstack((partial_features,ang_velocity_quat_arr))

    df[file_no] = partial_features

# Creating one-hot vector matrix
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


#%% =================================== Normalize Input data =================================== %%#
# ------------------------ Normalize input features based on thresholds set manually ----------------------------------#

# min and max values for each feature
#pos_x = (-9, 11)  # ideal +-28
pos_x = (-16,11)
pos_y = (-8, 20)  # ideal +-30
pos_z = (-42, -15)  # ideal +20 +60

vel_x = (0, 65)
vel_y = (0,70)
vel_z = (0,90)

ang_vel_x = (0,150)
ang_vel_y = (0,150)
ang_vel_z = (0,190)

#  input msx and min to normalize data
def normalize_dataframe_column_custom(min, max, input_df, feature_list):
    df_copy = input_df.copy()
    for feature in feature_list:
        df_copy[feature] = df_copy[feature].map(lambda a: normalize_input(a, min, max))
    return df_copy


def normalize_array_columns_custom(input_array, thresh_min: int, thresh_max: int):
    local_array = np.empty([input_array.size])
    # print('Size of array: ', input_array.size)
    for x in range(input_array.size):
        local_array[x] = normalize_input(input_array[x], thresh_min, thresh_max)
    return local_array


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

for file_no in range(len(df)):
    # normalize position and orientation values
    df[file_no] = normalize_dataframe_column_custom(pos_x[0], pos_x[1], df, ['X'])
    df[file_no] = normalize_dataframe_column_custom(pos_y[0], pos_y[1], df, ['Y'])
    df[file_no] = normalize_dataframe_column_custom(pos_z[0], pos_z[1], df, ['Z'])

    # normalize linear velocity
    x_norm_linear_velocity = normalize_array_columns_custom(x_linear_velocity, vel_x[0], vel_x[1])
    y_norm_linear_velocity = normalize_array_columns_custom(y_linear_velocity, vel_y[0], vel_y[1])
    z_norm_linear_velocity = normalize_array_columns_custom(z_linear_velocity, vel_z[0], vel_z[1])