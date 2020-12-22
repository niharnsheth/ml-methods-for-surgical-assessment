
# File imports and aggregates data from multiple databases
import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
from sqlite3  import Error


# File path to the database files
source_path = os.getcwd() + '/Data/'
data_folder = 'Six_Actions_Final/'
save_to_folder = 'Six_Actions_Updated/'



def add_padding(i, numpy_arr_1, numpy_arr_2):
    # for i in range(0,numpy_arr_long.shape[0]):
    #if numpy_arr_long[i,4] - numpy_arr_short[i,4] >= 0.02:

    return 0, np.insert(numpy_arr_1, i, np.array((numpy_arr_1[i-1,0],
                                             numpy_arr_1[i-1,1],
                                             numpy_arr_1[i-1,2],
                                             numpy_arr_1[i-1,3],
                                             numpy_arr_2[i,4])),0)


csv_list_1 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]
#file = "Set3_Action1.csv"


# read the file into a data-frame
for file_name in csv_list_1:
    print("File updating: {}".format(file_name))
    df = pd.read_csv(source_path + data_folder + file_name, header=None)
    # separate position and orientation data
    df_pos = df.iloc[:, [0, 1, 2, 3, 4]]
    df_pos = df_pos.rename(columns={0: 'SId', 1: 'X', 2: 'Y', 3: 'Z', 4: 'Pt'})
    df_ori = df.iloc[:, [0, 5, 6, 7, 8]]
    df_ori = df_ori.rename(columns={0: 'SId', 5: 'A', 6: 'B', 7: 'G', 8: 'Ot'})
    # Find Number of sensors used in task
    sensors_ids = df[0].unique()
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
        # Convert all to numpy_arrays for padding
        list_pos_numpy[i] = list_pos_df[i].to_numpy()
        list_ori_numpy[i] = list_ori_df[i].to_numpy()
        # Padding missing values to have equal length of POs and Ori
        padding_iter = 0
        while list_pos_numpy[i].shape[0] != list_ori_numpy[i].shape[0]:  # padding_iter <= max_len:
            if list_pos_numpy[i][padding_iter, 4] - list_ori_numpy[i][padding_iter, 4] >= 0.02:
                padding_iter, list_pos_numpy[i] = add_padding(padding_iter, list_pos_numpy[i], list_ori_numpy[i])
                # print("Number of pos rows: {}".format(numpy_pos.shape[0]))
            elif list_ori_numpy[i][padding_iter, 4] - list_pos_numpy[i][padding_iter, 4] >= 0.02:
                padding_iter, list_ori_numpy[i] = add_padding(padding_iter, list_ori_numpy[i], list_pos_numpy[i])
                # print("Number of ori rows: {}".format(numpy_ori.shape[0]))
            else:
                padding_iter = padding_iter + 1
                # print("Iterator: {}".format(padding_iter))

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

    # Create final dataset and save to csv
    total_no_sensors = 4
    features_per_sensor = 10
    total_features = total_no_sensors * features_per_sensor
    zero_value_numpy = 4 - list_length
    list_zero_numpy = [0 for i in range(zero_value_numpy)]
    for i in range(zero_value_numpy):
        list_zero_numpy[i] = np.zeros((longest_data_sequence_length,10))

    append_sequence = [0, 1, 2, 3]
    #final_numpy = np.empty((longest_data_sequence_index,total_features))
    final_numpy = np.zeros((longest_data_sequence_length,1))
    #sensor_sequence_index = 0
    sequence_index = 0
    sensor_index = 0
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

    column_delete_list = []
    final_numpy = np.delete(final_numpy,0,axis=1)
    for i in np.arange(0,40,5):
        print("Column deleting {}".format(i))
        column_delete_list = np.append(column_delete_list,i)

    final_numpy_test = np.delete(final_numpy,column_delete_list, axis=1)

    header_list = ['X1','Y1','Z1','PT1', 'A1','B1','G1','OT1',
                   'X2','Y2','Z2','PT2', 'A2','B2','G2','OT2',
                   'X3','Y3','Z3','PT3', 'A3','B3','G3','OT3',
                   'X4','Y4','Z4','PT4', 'A4','B4','G4','OT4',]
    final_df = pd.DataFrame(final_numpy_test, columns=header_list)
    final_df.to_csv(source_path + save_to_folder + file_name)
    #np.savetxt(source_path + save_to_folder + file_name, final_numpy_test, delimiter=',', header=header_list)
