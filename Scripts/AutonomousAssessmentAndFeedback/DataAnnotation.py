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
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
test_dir = os.listdir(source_path)
surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

# Get list of all data directories
# performance_list = os.listdir(source_path + surgery_name_list[surgery_selected] + '/')

sensor_id_list = ['0.csv', '1.csv', '2.csv', '3.csv']

data_folder_list = ['5 Actions_10032020/OriginalData/ChloraPrep/',
                    '5 Actions_10032020/OriginalData/NeedleInsertion/',
                    '5 Actions_10032020/OriginalData/Dilator/',
                    '5 Actions_10032020/OriginalData/Cut/',
                    '5 Actions_10032020/OriginalData/Tracing/']

# csv_list_1 = [f for f in os.listdir(source_path + surgery_name_list[0] ) if fnmatch.fnmatch(f, '*.csv')]


# --------------------------------------------------------------------------------------------------------- #
## ---------------------      Annotate the data based on experience    ------------------------ ##


input_folder = '/Normalization'
save_to_folder = '/Annotated'

# Get list of all data directories
test_file_path = source_path + input_folder + surgery_name_list[surgery_selected] + "/"
performance_list = os.listdir(test_file_path)


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

    # Create folder to save all the sensor files
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
        header_list = ["X", "Y", "Z", "W", "Qx", "Qy", "Qz",
                       "Vx", "Vy", "Vz", "VQx", "VQy", "VQz",
                       "Resident", "Fellow", "Expert"]

        df = np.hstack((df, label_df))
        df = pd.DataFrame(df)

        df.to_csv(source_path + save_to_folder +
                  surgery_name_list[surgery_selected] +
                  '/' + split_file_name[0] + '/' + individual_performance + '.csv',
                  index=False, header=header_list)
