# Script annotates and reduces noise of the original surgical data

## Order of processing
# Import --- Feature Extraction --- Threshold filtering --- Normalization

# --------------------------------------------------------------------------------------------------- #
## ---------------------      Import libraries and data       ------------------------ ##

# Run this cell first before running any other cells

import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import quaternion as qt
import time

# Select surgery: 0 or 1
# 0 - Pericardiocentesis
# 1 - Thoracentesis
surgery_selected = 0

# File path to the database files
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/TestData' # only to prep test data
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/ManuallyCleaned_06262021'
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/Manually_Cleaned_And_Annotated_06272021'
surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

# Get list of all data directories
#performance_list = os.listdir(source_path + surgery_name_list[surgery_selected] + '/')

sensor_id_list = ['0.csv', '1.csv', '2.csv', '3.csv']

# --------------------------------------------------------------------------------------------------------- #
## ---------------------      Extract features       ------------------------ ##
# Calculate linear and angular velocities

input_folder = '/OriginalData'
save_to_folder = '/ExtractedFeatures'


# Calculates linear and angular velocity for given values
def calculate_velocity(x1, x2, t1, t2):
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


# Get list of all data directories
performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')

for individual_performance in performance_list:

    # Get sensor data for each performance
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    # Create folder to save all the sensor files
    os.mkdir(source_path + save_to_folder + surgery_name_list[surgery_selected] +
             '/' + individual_performance)

    for data_sample in sensor_data:
        try:
            # read and import csv file
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] +
                             '/' + individual_performance +
                             '/' + data_sample)

        except pd.errors.EmptyDataError:
            continue

        # df = df.drop(columns=['SId','PT','OT'], axis=1)

        df = df.drop(columns=['SId'], axis=1)
        pos_time_stamps = df.loc[:, 'PT']
        pos_time_stamps = pos_time_stamps.values
        ori_time_stamps = df.loc[:, 'OT']
        ori_time_stamps = ori_time_stamps.values

        # calculate absolute linear velocities
        pos_values_array = df[['X', 'Y', 'Z']].to_numpy()
        # x_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 0], pos_time_stamps))
        # y_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 1], pos_time_stamps))
        # z_linear_velocity = np.absolute(cal_vel_for_range(pos_values_array[:, 2], pos_time_stamps))
        x_linear_velocity = cal_vel_for_range(pos_values_array[:, 0], pos_time_stamps)
        y_linear_velocity = cal_vel_for_range(pos_values_array[:, 1], pos_time_stamps)
        z_linear_velocity = cal_vel_for_range(pos_values_array[:, 2], pos_time_stamps)

        # convert euler angles to quaternion
        euler_angles_arr = df[['A', 'B', 'G']].to_numpy()
        np_quaternions_arr = np.empty([len(df.index), 4])
        for row in range(len(df.index)):
            np_quaternions_arr[row, :] = euler_to_quaternion(euler_angles_arr[row, 0],
                                                             euler_angles_arr[row, 1],
                                                             euler_angles_arr[row, 2])

        print("shape of quaternion array is: " + str(np_quaternions_arr.shape))
        # calculate the angular velocities
        quat_arr = qt.as_quat_array(np_quaternions_arr)
        print('Shape of converted quaternion is: ' + str(quat_arr.shape))
        ang_velocity_quat_arr = qt.quaternion_time_series.angular_velocity(quat_arr, ori_time_stamps)
        # ang_velocity_quat_arr = np.absolute(ang_velocity_quat_arr)

        final_features = np.hstack((pos_values_array,np_quaternions_arr))
        final_features = np.hstack((final_features,np.c_[x_linear_velocity]))
        final_features = np.hstack((final_features,np.c_[y_linear_velocity]))
        final_features = np.hstack((final_features,np.c_[z_linear_velocity]))
        final_features = np.hstack((final_features, ang_velocity_quat_arr))

        final_features = np.hstack((final_features,np.c_[pos_time_stamps]))
        final_features = np.hstack((final_features, np.c_[ori_time_stamps]))
        header_list = ["X", "Y", "Z", "W", "Qx", "Qy", "Qz", "Vx",
                       "Vy", "Vz", "VQx", "VQy", "VQz", "Pt", "Ot"]

        final_df = pd.DataFrame(final_features)
        final_df.to_csv(source_path + save_to_folder + surgery_name_list[surgery_selected] +
                        '/' + individual_performance + '/' + data_sample, index=False, header=header_list)



# --------------------------------------------------------------------------------------------------- #
## ---------------------      Filter data based on thresholds       ------------------------ ##

input_folder = '/ExtractedFeatures'
#input_folder = '/OriginalData'
save_to_folder = '/ThresholdFilter'

# Get list of all data directories
performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')

# Box dimensions
# 0 - pericardiocentesis   1 - thoracentesis
box_dimensions = [[200, 200, 200],
                  [200, 200, 200]]
box_positions = [[11.5, 58, -222],
                 [10.5, 99, -244]]

# old box position for thoracentesis
# [-2, 84.5, -220]

# variables
bounding_box_dimension = box_dimensions[surgery_selected]
bounding_box_position = box_positions[surgery_selected]
print("Bounding box values" + str(bounding_box_dimension[0]) + "," + str(bounding_box_dimension[1]) + "..")

bounding_box_dimension_from_center = [x / 2 for x in bounding_box_dimension]


# calculate the thresholds
def calc_thresholds(position_of_pivot, distance_from_pivot):
    min_threshold = position_of_pivot - distance_from_pivot
    max_threshold = position_of_pivot + distance_from_pivot
    return min_threshold, max_threshold


x_threshold = calc_thresholds(bounding_box_position[0], bounding_box_dimension_from_center[0])
y_threshold = calc_thresholds(bounding_box_position[1], bounding_box_dimension_from_center[1])
z_threshold = calc_thresholds(bounding_box_position[2], bounding_box_dimension_from_center[2])

# update data
for individual_performance in performance_list:
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    # Create folder to save all the sensor files
    os.mkdir(source_path + save_to_folder + surgery_name_list[surgery_selected] +
             '/' + individual_performance)

    for data_sample in sensor_data:
        try:
            # read sensor data csv
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] + '/'
                             + individual_performance + '/' + data_sample)

        except pd.errors.EmptyDataError:
            continue

        # only if file is not empty
        # filter x position based on threshold
        df_filter_threshold = df[df.X > x_threshold[0]]
        df_filter_threshold = df_filter_threshold[df_filter_threshold.X < x_threshold[1]]
        # filter x position based on threshold
        df_filter_threshold = df_filter_threshold[df_filter_threshold.Y > y_threshold[0]]
        df_filter_threshold = df_filter_threshold[df_filter_threshold.Y < y_threshold[1]]
        # filter x position based on threshold
        df_filter_threshold = df_filter_threshold[df_filter_threshold.Z > z_threshold[0]]
        df_filter_threshold = df_filter_threshold[df_filter_threshold.Z < z_threshold[1]]

        # reset the index
        # df_filter_threshold.reset_index(inplace=True)

        # save dataframe to csv file
        df_filter_threshold.to_csv(source_path + save_to_folder + surgery_name_list[surgery_selected] +
                                   '/' + individual_performance + '/' + data_sample, index=False)


# --------------------------------------------------------------------------------------------------------- #
## ---------------------      Normalize the data       ------------------------ ##

# input_folder = '/ManuallyAnnotatedForSiamese'
input_folder = '/ManuallyAnnotated'
# save_to_folder = '/TrainingDataForComparison'
save_to_folder = '/TrainingDataForClassification'
# save_to_folder = '/TestingDataForComparison'
# save_to_folder = '/TestingDataForClassification'

# Get list of all data directories
performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')

# min and max values for each features
# values obtained from plots. Plots stored in SurgicalData/Graph
# norm_threshold_peri = [Surgical action[X(min,max), Y.....]]
norm_thresholds_peri = [[(-25,75), (50,150), (-250,-160),
                        (-300,300),(-250,250), (-250,250),
                        (-75,75), (-75,75), (-75,75)],
                        [(0,50), (30,100), (-235,200),
                        (-100,80), (-90,50), (-70,90),
                        (-30,30),(-25,25), (-30,30)]]

norm_thresholds_thor = [[(-20,70),(80,135),(-260,-170),
                         (-200,200), (-150,150), (-200,200),
                         (-75,75), (-75,75), (-75,75)],
                        [(35,110),(132.5,200), (-317.5,-180),
                         (-150,150), (-90,100), (-100,100),
                         (-75,75), (-75,75), (-75,75)],
                        [(30,115), (70,185), (-325,-220),
                         (-90,90), (-75,75), (-90,90),
                         (-75,75), (-60,60), (-60,60)],
                        [(-35,100), (80,170), (-312.5, -165),
                         (-100,100), (-75,75), (-100,100),
                         (-100,100), (-100,100), (-100,100)]]


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


def normalize_input_1(x,x_min,x_max):
    if x >= x_max:
        return 1
    elif x <= x_min:
        return 0
    else:
        return 2 * ((x - x_min) / (x_max - x_min)) - 1


def scale_input(value, scale_value):
    if value > 0:
        return scale_value - value
    else:
        return scale_value + value


# use max and min values in sequence to normalize data
def normalize_column(input_df, feature_name, mode, min_value, max_value):
    df_copy = input_df.copy()

    for feature in feature_name:
        # max_value = input_df[feature].max()
        # min_value = input_df[feature].min()
        if max_value == min_value:
            print("Error: Cannot normalize when max and min values are equal")
            return df_copy
        elif mode == 0:
            df_copy[feature] = df_copy[feature].map(lambda a: normalize_input(a, min_value, max_value))
        elif mode == 1:
            df_copy[feature] = df_copy[feature].map(lambda a: normalize_input_1(a, min_value, max_value))
    return df_copy


## --------------       Normalize before Annotation         --------------------
for individual_performance in performance_list:
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    # Create folder to save all the sensor files
    os.mkdir(source_path + save_to_folder + surgery_name_list[surgery_selected] +
             '/' + individual_performance)

    for sample_ind in range(len(sensor_data)):
        try:
            # read sensor data csv
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] + '/'
                             + individual_performance + '/' + sensor_data[sample_ind])

        except pd.errors.EmptyDataError:
            continue

        # normalize position between custom thresholds, between 0 nad 1
        df = normalize_column(df, ['X'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][0][0],
                              max_value=norm_thresholds_thor[sample_ind][0][1])
        df = normalize_column(df, ['Y'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][1][0],
                              max_value=norm_thresholds_thor[sample_ind][1][1])
        df = normalize_column(df, ['Z'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][2][0],
                              max_value=norm_thresholds_thor[sample_ind][2][1])
        df = normalize_column(df, ['Vx'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][3][0],
                              max_value=norm_thresholds_thor[sample_ind][3][1])
        df = normalize_column(df, ['Vy'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][4][0],
                              max_value=norm_thresholds_thor[sample_ind][4][1])
        df = normalize_column(df, ['Vz'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][5][0],
                              max_value=norm_thresholds_thor[sample_ind][5][1])
        df = normalize_column(df, ['VQx'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][6][0],
                              max_value=norm_thresholds_thor[sample_ind][6][1])
        df = normalize_column(df, ['VQy'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][7][0],
                              max_value=norm_thresholds_thor[sample_ind][7][1])
        df = normalize_column(df, ['VQz'], mode=0,
                              min_value=norm_thresholds_thor[sample_ind][8][0],
                              max_value=norm_thresholds_thor[sample_ind][8][1])

        header_list = ["X", "Y", "Z", "W", "Qx", "Qy", "Qz", "Vx",
                       "Vy", "Vz", "VQx", "VQy", "VQz", "Pt", "Ot"]

        df = pd.DataFrame(df)
        df.to_csv(source_path + save_to_folder + surgery_name_list[surgery_selected] +
                  '/' + individual_performance + '/' + sensor_data[sample_ind], index=False, header=header_list)

##  ---------------------           Normalize after annotation            -----------------------

for surgical_task_idx in range(len(performance_list)):
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + performance_list[surgical_task_idx] + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    # Create folder to save all the sensor files
    os.mkdir(source_path + save_to_folder + surgery_name_list[surgery_selected] +
             '/' + performance_list[surgical_task_idx])

    for sample_ind in range(len(sensor_data)):
        try:
            # read sensor data csv
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] + '/'
                             + performance_list[surgical_task_idx] + '/' + sensor_data[sample_ind])

        except pd.errors.EmptyDataError:
            continue

        # drop the time columns
        df.drop(['Pt','Ot'], axis=1, inplace=True)

        # normalize position between custom thresholds, between 0 nad 1
        df = normalize_column(df, ['X'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][0][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][0][1])
        df = normalize_column(df, ['Y'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][1][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][1][1])
        df = normalize_column(df, ['Z'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][2][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][2][1])
        df = normalize_column(df, ['Vx'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][3][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][3][1])
        df = normalize_column(df, ['Vy'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][4][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][4][1])
        df = normalize_column(df, ['Vz'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][5][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][5][1])
        df = normalize_column(df, ['VQx'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][6][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][6][1])
        df = normalize_column(df, ['VQy'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][7][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][7][1])
        df = normalize_column(df, ['VQz'], mode=0,
                              min_value=norm_thresholds_thor[surgical_task_idx][8][0],
                              max_value=norm_thresholds_thor[surgical_task_idx][8][1])

        header_list = ["X", "Y", "Z",
                       "W", "Qx", "Qy", "Qz",
                       "Vx", "Vy", "Vz",
                       "VQx", "VQy", "VQz",
                       "Novice", "Intermediate", "Expert"]
        # header_list = ["X", "Y", "Z",
        #                "W", "Qx", "Qy", "Qz",
        #                "Vx", "Vy", "Vz",
        #                "VQx", "VQy", "VQz",
        #                "Similarity"]

        df = pd.DataFrame(df)
        df.to_csv(source_path + save_to_folder + surgery_name_list[surgery_selected] +
                  '/' + performance_list[surgical_task_idx] + '/' +
                  sensor_data[sample_ind], index=False, header=header_list)


##

