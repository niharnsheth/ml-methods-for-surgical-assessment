# File imports and aggregates data from multiple databases
import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
from sqlite3 import Error

# File path to the database files
source_path = os.getcwd() + '/Data/'
data_folder = 'Six_Actions_Original/'
save_to_folder = 'Six_Actions_Updated/'


def calculate_timeframes(max_obs, del_t):
    total_obs = max_obs * del_t
    # print("total obs: {}".format(total_obs))
    if total_obs < 1:
        return 1
    else:
        return 1 / total_obs


def single_lerp(value_a, value_b, ratio):
    ret_val = (ratio * value_a) + ((1 - ratio) * value_b)
    # print("Value returned: {}".format(ret_val))
    return ret_val


def vector_lerp(_a, _b, ratio):
    return_vec = [0, 0, 0, 0]
    for i in range(len(_a)):
        return_vec[i] = single_lerp(_a[i], _b[i], ratio)
    return return_vec


def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def decompress(numpy_array, sensor_id):
    uncomp_list = [[], [], [], [], []]
    # Decompressing position data
    for j in range(len(numpy_array) - 1):
        del_time = numpy_array[j + 1, 4] - numpy_array[j, 4]
        # print("Delta time: {}".format(del_time))
        increment_ratio = calculate_timeframes(50, del_time)
        increment_values = increment_ratio
        # print("Increment by: {}".format(increment_ratio))
        vec_a = numpy_array[j + 1, 1:5]
        vec_b = numpy_array[j, 1:5]
        # k = 0
        while increment_values < 1.02:
            missing_vec_pos = vector_lerp(vec_a, vec_b, increment_values)
            print()
            # print("Time data: {}".format(missing_vec_pos[3]))
            uncomp_list[0].append(sensor_id)
            uncomp_list[1].append(missing_vec_pos[0])
            uncomp_list[2].append(missing_vec_pos[1])
            uncomp_list[3].append(missing_vec_pos[2])
            uncomp_list[4].append(missing_vec_pos[3])
            increment_values += increment_ratio
            # k += 1
        # print("Total added: {}".format(k))
    return uncomp_list


csv_list_1 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]

column_names = ['X0', 'Y0', 'Z0', 'Tp0', 'A0', 'B0', 'G0', 'To0',
                'X1', 'Y1', 'Z1', 'Tp1', 'A1', 'B1', 'G1', 'To1',
                'X2', 'Y2', 'Z2', 'Tp2', 'A2', 'B2', 'G2', 'To2',
                'X3', 'Y3', 'Z3', 'Tp3', 'A3', 'B3', 'G3', 'To3']

##  ---   Normalize the data  --- #
# min and max values for each feature
pos_x = (25, -18)  # ideal +-28
pos_y = (10, -15)  # ideal +-30
pos_z = (-12, -40)  # ideal +20 +60
ang_x = (90, -90)
ang_y = (180, -180)
ang_z = (180, -180)


#  input msx and min to normalize data
def normalize_column_custom(max, min, input_df, feature_list):
    df_copy = input_df.copy()
    for feature in feature_list:
        df_copy[feature] = df_copy[feature].map(lambda a: scale_input(a, max, min))
        # df_copy[feature] = (input_df[feature] - min) / (max - min)
    return df_copy


def scale_input(x, max, min):
    if x >= max:
        return 1
    elif x <= min:
        return 0
    else:
        return (x - min) / (max - min)


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


## Clean and Normalize data
for file in csv_list_1:
    print("File updating: {}".format(file))
    df = pd.read_csv(source_path + data_folder + file, header=None)
    # separate position and orientation data
    df_pos = df.iloc[:, [0, 1, 2, 3, 4]]
    df_pos = df_pos.rename(columns={0: 'SId', 1: 'X', 2: 'Y', 3: 'Z', 4: 'Pt'})
    df_ori = df.iloc[:, [0, 5, 6, 7, 8]]
    df_ori = df_ori.rename(columns={0: 'SId', 5: 'A', 6: 'B', 7: 'G', 8: 'Ot'})
    # Find Number of sensors used in task
    total_sensor_ids = [0, 1, 2, 3]
    sensors_ids = df[0].unique()
    sensors_ids = np.sort(sensors_ids)
    list_length = len(sensors_ids)
    pos_df = pd.DataFrame()
    ori_df = pd.DataFrame()
    # list_pos_numpy = [0 for i in range(len(sensors_ids))]
    # list_ori_numpy = [0 for i in range(len(sensors_ids))]
    print("Sensors used: {}".format(sensors_ids))
    uncomp_dfs = [pd.DataFrame() for i in range(len(total_sensor_ids))]

    # Traverse through each dataframe separated based on sensorId, to prep data
    for i in total_sensor_ids:
        if i in sensors_ids:
            # Separate data based on SensorId before padding
            list_pos_df = df_pos.loc[df_pos['SId'] == i]
            list_ori_df = df_ori.loc[df_ori['SId'] == i]
            # Remove all Nan value rows
            clean_pos_df = list_pos_df.dropna(thresh=3)
            clean_ori_df = list_ori_df.dropna(thresh=3)

            clean_pos_df.reset_index(drop=True, inplace=True)
            clean_ori_df.reset_index(drop=True, inplace=True)
            # Convert all to numpy_arrays for decompression
            list_pos_numpy = clean_pos_df.to_numpy()
            list_ori_numpy = clean_ori_df.to_numpy()
            # Decompressing position data
            uncomp_pos_list = decompress(list_pos_numpy, i)
            uncomp_ori_list = decompress(list_ori_numpy, i)

            # uncomp_numpy = np.concatenate(uncomp_pos_list, uncomp_ori_list, axis=1)
            uncomp_pos_df = pd.DataFrame(data=uncomp_pos_list[:][1:]).T
            uncomp_ori_df = pd.DataFrame(data=uncomp_ori_list[:][1:]).T

            uncomp_pos_df = normalize_column_custom(pos_x[0], pos_x[1], uncomp_pos_df, [0])
            uncomp_pos_df = normalize_column_custom(pos_y[0], pos_y[1], uncomp_pos_df, [1])
            uncomp_pos_df = normalize_column_custom(pos_z[0], pos_z[1], uncomp_pos_df, [2])
            uncomp_ori_df = normalize_column_custom(ang_x[0], ang_x[1], uncomp_ori_df, [0])
            uncomp_ori_df = normalize_column_custom(ang_y[0], ang_y[1], uncomp_ori_df, [1])
            uncomp_ori_df = normalize_column_custom(ang_z[0], ang_z[1], uncomp_ori_df, [2])

            uncomp_dfs[i] = pd.concat([uncomp_pos_df, uncomp_ori_df], axis=1)
        else:
            uncomp_dfs[i] = pd.DataFrame(np.zeros((1, 8)))

    final_df = pd.DataFrame()

    for i in range(len(total_sensor_ids)):
        final_df = pd.concat([final_df, uncomp_dfs[i]], axis=1)
        # final_df.fillna(0, inplace=True)
    final_df.fillna(0, inplace=True)
    final_df.columns = column_names
    task_no = int(file[-5])
    one_hot_df = pd.DataFrame(np.zeros((final_df.shape[0], 6)))
    one_hot_df[task_no - 1] = 1
    final_df = pd.concat([final_df, one_hot_df], axis=1)
    final_df.to_csv(source_path + save_to_folder + file)
