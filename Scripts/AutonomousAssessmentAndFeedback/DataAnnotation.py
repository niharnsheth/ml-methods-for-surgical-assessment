# Script annotates normalized and clean data

## Order of processing
# Import --- Annotation

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

# File path to the database files
#source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/TestData' # only to prep test data
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/ManuallyCleaned_06262021'
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/Manually_Cleaned_And_Annotated_06272021'
surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

# Get list of all data directories
# performance_list = os.listdir(source_path + surgery_name_list[surgery_selected] + '/')

sensor_id_list = ['0.csv', '1.csv', '2.csv', '3.csv']

data_labels = ['Resident', 'Fellow', 'Expert']
# csv_list_1 = [f for f in os.listdir(source_path + surgery_name_list[0] ) if fnmatch.fnmatch(f, '*.csv')]


# --------------------------------------------------------------------------------------------------------- #
## ---------------------      Annotate the data based on experience    ------------------------ ##


input_folder = '/Normalized'
save_to_folder = '/ManuallyAnnotated'

# Get list of all data directories
# test_file_path = source_path + input_folder + surgery_name_list[surgery_selected] + "/"
# performance_list = os.listdir(test_file_path)
performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + "/")


# Create labels
def create_label_df(my_df, number_of_actions, action_index):
    # Create one-hot vector labels df
    num_of_rows = len(my_df.index)
    labels_arr = np.zeros((num_of_rows, number_of_actions))
    labels_arr[:, action_index] = 1
    return labels_arr


# Return index for annotation
def check_experience_level(experience):
    if fnmatch.fnmatch(experience, 'Resident'):
        return 0
    elif fnmatch.fnmatch(experience, 'Fellow'):
        return 1
    else:
        return 2


# update data
for individual_performance in performance_list:
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    # # Create folder to save all the sensor files
    # os.mkdir(source_path + save_to_folder + surgery_name_list[surgery_selected] +
    #          '/' + individual_performance)

    # Get experience level
    split_list = individual_performance.split('_')
    experience_level = split_list[1]
    exp_index = check_experience_level(experience_level)

    for data_sample in sensor_data:
        try:
            # read sensor data csv
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] + '/'
                             + individual_performance + '/' + data_sample)

        except pd.errors.EmptyDataError:
            continue

        # drop time columns
        df = df.drop(['Pt','Ot'], axis=1)
        # create a df of labels
        label_df = create_label_df(df, 3, exp_index)
        # get the sensor Id to store the data based on action performed
        split_file_name = data_sample.split('.')

        # update header list
        # header_list = ["X", "Y", "Z", "W", "Qx", "Qy", "Qz",
        #                "Vx", "Vy", "Vz", "VQx", "VQy", "VQz",
        #                "Pt", "Ot", "Resident", "Fellow", "Expert"]

        header_list = ["X", "Y", "Z", "W", "Qx", "Qy", "Qz",
                       "Vx", "Vy", "Vz", "VQx", "VQy", "VQz",
                       "Resident", "Fellow", "Expert"]

        df = np.hstack((df, label_df))
        df = pd.DataFrame(df)

        df.to_csv(source_path + save_to_folder +
                  surgery_name_list[surgery_selected] +
                  '/' + split_file_name[0] + '/' + individual_performance + '.csv',
                  index=False, header=header_list)


# --------------------------------------------------------------------------------------------------------- #
## ---------------------      Manually Annotate the data based on experience    ------------------------ ##


input_folder = '/ThresholdFilter'
save_to_folder = '/ManuallyAnnotated'
# ref_file = '/ReferenceAnnotation.csv'

performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + "/")


# Create labels
def create_label_df(my_df, number_of_actions, action_index):
    # Create one-hot vector labels df
    num_of_rows = len(my_df.index)
    labels_arr = np.zeros((num_of_rows, number_of_actions))
    labels_arr[:, action_index] = 1
    return labels_arr


# Return index for annotation
def check_experience_level(experience):
    if fnmatch.fnmatch(experience, 'Novice'):
        return 0
    elif fnmatch.fnmatch(experience, 'Intermediate'):
        return 1
    elif fnmatch.fnmatch(experience, 'Expert'):
        return 2
    else:
        return 3


manually_annotated_labels = pd.read_csv(source_path + "/" + surgery_name_list[surgery_selected][1:] + ".csv")

# update data
for individual_performance in performance_list:
    # get data collected from a sensor
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    # Get experience level
    file_idx = manually_annotated_labels.index[manually_annotated_labels['PerformanceName'] ==
                                               individual_performance]

    for data_sample in sensor_data:
        try:
            # read sensor data csv
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] + '/'
                             + individual_performance + '/' + data_sample)

        except pd.errors.EmptyDataError:
            continue

        # drop time columns
        # df = df.drop(['Pt', 'Ot'], axis=1)
        # get label
        split_list = data_sample.split('.')
        # get the sensor Id to store the data based on action performed
        split_file_name = data_sample.split('.')
        experience_level = manually_annotated_labels.iloc[file_idx][split_file_name[0]].iloc[0]
        exp_index = check_experience_level(experience_level)

        if exp_index == 3:
            break
        # create a df of labels
        label_df = create_label_df(df, 3, exp_index)

        # update header list
        header_list = ["X", "Y", "Z", "W", "Qx", "Qy", "Qz",
                       "Vx", "Vy", "Vz", "VQx", "VQy", "VQz",
                       "Pt", "Ot", "Resident", "Fellow", "Expert"]

        df = np.hstack((df, label_df))
        df = pd.DataFrame(df)

        df.to_csv(source_path + save_to_folder +
                  surgery_name_list[surgery_selected] +
                  '/' + split_file_name[0] + '/' + individual_performance + '.csv',
                  index=False, header=header_list)

