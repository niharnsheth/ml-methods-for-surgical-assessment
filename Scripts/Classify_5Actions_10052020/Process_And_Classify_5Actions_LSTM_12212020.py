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

for i in range(len(data_folder_list)):
    # get list of samples for action
    action_no = i
    csv_list_1 = [f for f in os.listdir(source_path + data_folder_list[i]) if fnmatch.fnmatch(f, '*.csv')]
    # read and import csv files
    for file in csv_list_1:
        print("File updating: {}".format(file))
        # read data from csv into a dataframe
        file_df = pd.read_csv(source_path + data_folder_list[i] + file)
        df.append(file_df.drop(columns=['SId'], axis=1))


#%% ============================================== Add Features ==================================================== %%#
# ---------------------------------- Add linear and angular velocities to input data ----------------------------------#

columns_to_add = ['Vx', 'Vy', 'Vz', 'Va', 'Vb', 'Vg']


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


#for file in df:
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

    partial_features = np.hstack((pos_values_array[:,0:2],np_quaternions_arr))
    partial_features = np.hstack((partial_features,np.c_[x_linear_velocity],
                                  np.c_[y_linear_velocity],
                                  np.c_[z_linear_velocity]))
    partial_features = np.hstack((partial_features,ang_velocity_quat_arr))

    df[file_no] = partial_features