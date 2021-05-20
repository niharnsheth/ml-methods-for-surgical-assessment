# Script segments data creates motion windows and segments random selected windows based on experience


# --------------------------------------------------------------------------------------------------- #
## ---------------------      Import libraries and data       ------------------------ ##

# Run this cell first before running any other cells

import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import quaternion as qt

# Select surgery: 0 or 1
# 0 - Pericardiocentesis
# 1 - Thoracentesis
surgery_selected = 0
sliding_window_size = 150

# File path to the database files
# source_path = os.getcwd() + '/../..' + '/Data/Data_04152021'
source_path = os.getcwd() + '/Data/Data_04152021'
# save_to_folder = '/ThresholdFilter'

surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

# Get list of all data directories
# performance_list = os.listdir(source_path + surgery_name_list[surgery_selected] + '/')

sensor_id_list = ['0.csv', '1.csv', '2.csv', '3.csv']

# csv_list_1 = [f for f in os.listdir(source_path + surgery_name_list[0] ) if fnmatch.fnmatch(f, '*.csv')]

# --------------------------------------------------------------------------------------------------------- #
## ---------------------      Create motion windows       ------------------------ ##
# Create motion windows and save based on experience level

input_folder = '/Annotated'
save_to_folder = '/SegmentedData'


# return an input and output data frame with motion windows
def create_and_save_motion_windows(window_size, df_to_change, window_file_path, file_name):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - window_size)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step + window_size, :-2].reset_index(drop=True)
        a.to_csv(window_file_path + "/Features" + "/" + file_name + "_" + str(step) + ".csv", index=False)
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step + window_size, 13:].reset_index(drop=True)
        b.to_csv(window_file_path + "/Labels" + "/" + file_name + "_" + str(step) + ".csv", index=False)
        #local_feature_df.append(a)
        #local_label_df.append(b)
    #return local_feature_df, local_label_df
    return "done"


feature_list = list()
label_list = list()

# Get list of all data directories
performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')

for individual_performance in performance_list:

    split_list = individual_performance.split('_')
    experience_level = split_list[1]

    # Get sensor data for each performance
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    # Create folder to save all the sensor files
    # os.mkdir(source_path + save_to_folder + surgery_name_list[surgery_selected] +
    #          '/' + individual_performance)

    for data_sample in sensor_data:
        try:
            # read and import csv file
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] +
                             '/' + individual_performance +
                             '/' + data_sample)

        except pd.errors.EmptyDataError:
            continue

        # Break down data to windows
        create_and_save_motion_windows(sliding_window_size,
                                       df,
                                       source_path + save_to_folder + surgery_name_list[surgery_selected]
                                       + "/" + experience_level,
                                       individual_performance)
