# Script creates visual representations of data to determine nomalization parameters and
# appropriate filters for cleaning data

# --------------------------------------------------------------------------------------------------- #
## ---------------------      Import libraries and data       ------------------------ ##

# Run this cell first before running any other cells

import os
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Select surgery: 0 or 1
# 0 - Pericardiocentesis
# 1 - Thoracentesis
surgery_selected = 1

# File path to the database files
#source_path = os.getcwd() + '/../..' + '/Data/Data_04152021'
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/Manually_Cleaned_And_Annotated_06272021'
#save_to_folder = '/ThresholdFilter'

surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

# Get list of all data directories
#performance_list = os.listdir(source_path + surgery_name_list[surgery_selected] + '/')

sensor_id_list = ['0.csv', '1.csv', '2.csv', '3.csv']

# Determine the thresholds for normalization

## Import data for visualization
# obtain list of files

input_folder = '/ThresholdFilter'
save_to_folder = '/DataVisualization'


column_name_list = ['X', 'Y', 'Z', 'W', 'Qx', 'Qy', 'Qz', 'Vx', 'Vy', 'Vz', 'VQx', 'VQy', 'VQz']
data_type_list = ['Position', 'Orientation', 'Linear_Velocity', 'Angular_Velocity']
action_list = ['Chloraprep', 'Scalpel_Incision', 'Trocar_Insertion', 'Anesthetizing']
# action_list = ['Chloraprep', 'Needle_Insertion']
action_legends = [0,0,0]
fig, ax1 = plt.subplots()

main_df = pd.DataFrame()
seq_df = pd.DataFrame()

column_index = 3
column_name = column_name_list[3]
data_type = data_type_list[1] # type of data, position, velocity
action_selected = action_list[3] # surgical action
sensor_id = 3  # sensor Id

# Get list of all data directories
performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')
for individual_performance in performance_list:
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, str(sensor_id) + '.csv')]

    for data_sample in sensor_data:
        try:
            # read sensor data csv
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] + '/'
                             + individual_performance + '/' + data_sample)

        except pd.errors.EmptyDataError:
            continue

        # removing index column
        # df = df.drop(columns=['SId'], axis=1)
        # get length of data
        length_of_df = df.shape[0]
        # create a sequence of numbers for plotting
        # seq_df = pd.DataFrame(range(0, length_of_df))
        # seq_df = seq_df.append(pd.DataFrame(range(0, length_of_df)), ignore_index=True)
        # attach data to
        seq_df = seq_df.append(pd.DataFrame(range(0, length_of_df)), ignore_index=True)
        main_df = main_df.append(df, ignore_index=True)
        # ax1.scatter(main_np_arr[:,8],main_np_arr[:,4], label=labels_legend[i])


main_np_arr = main_df.to_numpy()
seq_np_arr = seq_df.to_numpy()
plt.hist(main_np_arr[:,column_index], color='blue', edgecolor='black', bins= 50 )
# ax1.scatter(main_np_arr[:, column_index], seq_np_arr[:, 0], s=5)
# fig = plt.figure()
# ax = plt.axes(projection='3d')

plt.title(surgery_name_list[surgery_selected][1:] + ': ' + action_selected +
          '- ' + column_name + ' -' + data_type + ' values')
plt.xlabel(data_type)
plt.ylabel('Counts')
# plt.xlim([-150,150])
# plt.ylim([0,1250])
# plt.xticks(np.arange(-150,150,25))
plt.savefig(source_path + save_to_folder + '/' +
            surgery_name_list[surgery_selected][1:] + '_' + str(action_selected) +
            '_' + column_name + '_' + data_type + '.png',
            dpi=300, bbox_inches='tight')
plt.show()

# main_np_arr = main_df.to_numpy()
# seq_np_arr = seq_df.to_numpy()


## Plotting number of windows
# Create and count number of windows for each class
input_folder = '/Segmented_AutoAnnotated/Pericardiocentesis/2'
class_list = os.listdir(source_path + input_folder)
window_size = 150
step_size = [2,5,1]


# return an input and output data frame with motion windows
def create_motion_windows(window_span, df_to_change, step_size):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - window_span)
    time_index = 0
    while time_index + window_span < len(df_to_change):
        a = df_to_change.iloc[time_index:time_index + window_span, :-1].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[time_index + window_span, 13:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
        time_index += step_size
    return local_feature_df, local_label_df


for label_ind in range(len(class_list)):
    sample_list = [f for f in os.listdir(source_path + input_folder + '/' + class_list[label_ind])
                   if fnmatch.fnmatch(f, '*.csv')]
    feature_list_main = []
    label_list_main = []
    stp_sz = step_size[label_ind]
    for sample in sample_list:
        try:
            df = pd.read_csv(source_path + input_folder + '/' + class_list[label_ind] + '/' + sample)
        except pd.errors.EmptyDataError:
            continue
        # create motion windows and separate data into input and output
        # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)w
        feature_list, label_list = create_motion_windows(window_size, df, stp_sz)
        # create list of windows
        feature_list_main.extend(feature_list)
        label_list_main.extend(label_list)

    print("Class: " + class_list[label_ind])
    print(len(feature_list_main))

## Test me

norm_thresholds = [[(-25,75), (50,100), (-260,-160),
                   (-300,300),(-150,150), (-250,250),
                   (-75,75), (-75,75), (-75,75)],
                   [(0,50), (30,100), (-235,200),
                    (-90,80), (-90,50), (-70,90),
                    (-30,30),(-25,25), (-30,30)]]

print(norm_thresholds[1][1][0])

## Test me 2

feature_list= [2,3,4,5,21,1,4,2,5,6,4]
print(len(feature_list))
rand_list = np.random.choice(feature_list, 5)
print(rand_list)

## Test 2

start = time.time()
for i in range(100):
    print(str(i))
end = time.time()
print(end - start)

## Test 3

str_me = '0,1,2,3,4,5'
lst = str(str_me).split(',')
lst = [float(x) for x in lst[1:]]
lst[0] = int(lst[0])


cur_time= time.time()
print(cur_time)
lst.append(cur_time)

print(len(lst))
print(lst)

## Test 4
excess_samples = 5
for i in range(excess_samples):
    print(str(i))
