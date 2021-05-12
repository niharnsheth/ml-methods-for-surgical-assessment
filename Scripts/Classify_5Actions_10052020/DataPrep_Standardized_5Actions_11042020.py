# Script prepares data for training. Missing columns for sensors are added.

import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import quaternion as qt

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
    return [quat_w, quat_x, quat_y, quat_z]

## Calculate velocity features


columns_to_add = ['Vx', 'Vy', 'Vz', 'Va', 'Vb', 'Vg']


# Calculates linear and angular velocity for given values
def calculate_velocity(x1,x2,t1,t2):
    return (x2 - x1) / (t2 - t1)


# Calculate velocity for a list of consecutive values
def cal_vel_for_range(pos_ori_values, time_values):
    # Check length of lists
    if len(pos_ori_values) == len(time_values):
        # initialize local variables
        velocity = []
        velocity.append(0)
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


# Calculate velocity for multiple lists in a cvs file
def cal_vel_for_file(path_to_file, name_of_file):
    # Read in the file data into a dataframe
    df_local = pd.read_csv(path_to_file+name_of_file)
    # Check if time column exits
    if 'T' in df_local.columns:
        # Delete the index column
        df_local.drop('r', axis=1, inplace=True)
        # store the the column headers
        current_columns = df_local.columns
        print(current_columns)
        print("Number of colums are: {}".format(len(current_columns)))
        #print(current_columns)
        column_index = 0
        # Loop through each column through consecutive values to calculate velocity
        while column_index < len(current_columns) - 1:
            print("column_id = {}".format(column_index))
            velocity_list = cal_vel_for_range(df_local[current_columns[column_index]], df_local['T'])
            velocity_list.insert(0, 0)
            df_local.insert(len(df_local.columns), columns_to_add[column_index], velocity_list)
            column_index += 1
    return df_local


## Create labels
def create_label_df(my_df, number_of_actions, action_index):
    # Create one-hot vector labels df
    num_of_rows = len(my_df.index)
    labels_arr = np.zeros((num_of_rows, number_of_actions))
    labels_arr[:, action_index] = 1
    return labels_arr


## Clean and Normalize data
for i in range(len(data_folder_list)):
    # get list of samples for action
    action_no = i
    csv_list_1 = [f for f in os.listdir(source_path + data_folder_list[i]) if fnmatch.fnmatch(f, '*.csv')]
    for file in csv_list_1:

        print("File updating: {}".format(file))
        # read data from csv into a dataframe
        # read and import csv file
        df = pd.read_csv(source_path + data_folder_list[i] + file)
        #df = df.drop(columns=['SId','PT','OT'], axis=1)

        df = df.drop(columns=['SId'], axis=1)
        pos_time_stamps = df.loc[:,'PT']
        pos_time_stamps = pos_time_stamps.values
        ori_time_stamps = df.loc[:,'OT']
        ori_time_stamps = ori_time_stamps.values

        # calculate absolute linear velocities
        pos_values_array = df[['X', 'Y', 'Z']].to_numpy()
        x_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 0], pos_time_stamps))
        y_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 1], pos_time_stamps))
        z_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 2], pos_time_stamps))

        # normalize position
        df = normalize_dataframe_column_custom(pos_x[0], pos_x[1], df, ['X'])
        df = normalize_dataframe_column_custom(pos_y[0], pos_y[1], df, ['Y'])
        df = normalize_dataframe_column_custom(pos_z[0], pos_z[1], df, ['Z'])

        # normalize linear velocity
        x_norm_linear_velocity = normalize_array_columns_custom(x_linear_velocity, vel_x[0], vel_x[1])
        y_norm_linear_velocity = normalize_array_columns_custom(y_linear_velocity, vel_y[0], vel_y[1])
        z_norm_linear_velocity = normalize_array_columns_custom(z_linear_velocity, vel_z[0], vel_z[1])
        # vectorized_func = np.vectorize(normalize_input)
        # x_norm_linear_velocity = vectorized_func(x_linear_velocity,vel_x[0], vel_x[1])
        # y_norm_linear_velocity = vectorized_func(y_linear_velocity,vel_y[0], vel_y[1])
        # z_norm_linear_velocity = vectorized_func(z_linear_velocity,vel_z[0], vel_z[1])

        # Get the normalized values
        pos_values_array = df[['X', 'Y', 'Z']].to_numpy()

        # convert euler angles to quaternion
        euler_angles_arr = df[['A','B','G']].to_numpy()
        np_quaternions_arr = np.empty([len(df.index),4])
        for row in range(len(df.index)):
            np_quaternions_arr[row,:] = euler_to_quaternion(euler_angles_arr[row,0],
                                                            euler_angles_arr[row,1],
                                                            euler_angles_arr[row,2])


        # calculate the angular velocities
        quat_arr = qt.as_quat_array(np_quaternions_arr)
        ang_velocity_quat_arr = qt.quaternion_time_series.angular_velocity(quat_arr,ori_time_stamps)
        ang_velocity_quat_arr = np.absolute(ang_velocity_quat_arr)

        # normalize angular velocities
        norm_angular_velocity_quat_arr = ang_velocity_quat_arr
        norm_angular_velocity_quat_arr[:,0] = normalize_array_columns_custom(ang_velocity_quat_arr[:,0],
                                                                             ang_vel_x[0], ang_vel_x[1])
        norm_angular_velocity_quat_arr[:,1] = normalize_array_columns_custom(norm_angular_velocity_quat_arr[:, 1],
                                                                             ang_vel_y[0], ang_vel_y[1])
        norm_angular_velocity_quat_arr[:,2] = normalize_array_columns_custom(norm_angular_velocity_quat_arr[:, 2],
                                                                             ang_vel_z[0], ang_vel_z[1])

        # # calculate linear velocities
        # pos_values_array = df[['X','Y','Z']].to_numpy()
        # x_vel = cal_vel_for_range(pos_values_array[:,0], pos_time_stamps)
        # y_vel = cal_vel_for_range(pos_values_array[:,1], pos_time_stamps)
        # z_vel = cal_vel_for_range(pos_values_array[:,2], pos_time_stamps)

        final_features = np.hstack((pos_values_array, np_quaternions_arr))
        final_features = np.hstack((final_features, np.c_[x_norm_linear_velocity]))
        final_features = np.hstack((final_features, np.c_[y_norm_linear_velocity]))
        final_features = np.hstack((final_features, np.c_[z_norm_linear_velocity]))
        final_features = np.hstack((final_features, norm_angular_velocity_quat_arr))


        # dataframe of one-hot-vectors
        final_labels = create_label_df(df, no_of_actions, action_no)
        processed_features_and_labels = np.hstack((final_features, final_labels))

        header_list = ['X1', 'Y1', 'Z1', 'PT1', 'A1', 'B1', 'G1', 'OT1',
                       'X2', 'Y2', 'Z2', 'PT2', 'A2', 'B2', 'G2', 'OT2']

        final_df = pd.DataFrame(processed_features_and_labels)
        final_df.to_csv(source_path + save_to_folder + file)