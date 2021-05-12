# Script creates visual representations of data to determine nomalization parameters and
# appropriate filters for cleaning data

# --------------------------------------------------------------------------------------------------- #
## ---------------------      Import libraries and data       ------------------------ ##

# Run this cell first before running any other cells

import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import quaternion as qt
import matplotlib.pyplot as plt

# Select surgery: 0 or 1
# 0 - Pericardiocentesis
# 1 - Thoracentesis
surgery_selected = 0

# File path to the database files
#source_path = os.getcwd() + '/../..' + '/Data/Data_04152021'
source_path = os.getcwd() + '/Data/Data_04152021'
#save_to_folder = '/ThresholdFilter'

surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

# Get list of all data directories
#performance_list = os.listdir(source_path + surgery_name_list[surgery_selected] + '/')

sensor_id_list = ['0.csv', '1.csv', '2.csv', '3.csv']

# Determine the thresholds for normalization

## Plot data to find range of normalization
# obtain list of files

input_folder = '/OriginalData'
save_to_folder = '/Normalization'

# Get list of all data directories
performance_list = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')

labels_legend = ['ChloraPrep', 'NeedleInsert', 'Dilator', 'Cut', 'Tracing']
action_legends = [0,0,0]
fig, ax1 = plt.subplots()

main_df = pd.DataFrame()
seq_df = pd.DataFrame()

# update data
for individual_performance in performance_list:
    sensor_data = [f for f in os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] +
                                         '/' + individual_performance + '/')
                   if fnmatch.fnmatch(f, '*.csv')]

    for data_sample in sensor_data:
        try:
            # read sensor data csv
            df = pd.read_csv(source_path + input_folder + surgery_name_list[surgery_selected] + '/'
                             + individual_performance + '/' + data_sample)

        except pd.errors.EmptyDataError:
            continue

        length_of_df = df.shape[0]
        seq_df = seq_df.append(pd.DataFrame(range(0, length_of_df)), ignore_index=True)
        main_df = main_df.append(df, ignore_index=True)


main_np_arr = main_df.to_numpy()
seq_np_arr = seq_df.to_numpy()
# ax1.scatter(main_np_arr[:,8],main_np_arr[:,4], label=labels_legend[i])
ax1.scatter(main_np_arr[:, 3], seq_np_arr[:, 0], s=5)

#plt.legend()
plt.xlim([-400,400])
plt.xticks(np.arange(-400,400,50))

plt.title('Z positional values')
plt.xlabel('z_values')
plt.ylabel('time_steps')
plt.show()
