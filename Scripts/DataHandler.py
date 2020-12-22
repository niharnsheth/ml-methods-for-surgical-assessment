# File imports and aggregates data from multiple databases
import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
from sqlite3  import Error


# File path to the database files
source_path = os.getcwd() + '/Data/'
database_folder = 'Circle_Square/'
action_time_csv = 'DataTrialTime.csv'
square_data_folder = 'Square_CSV/'
circle_data_folder = 'Circle_CSV/'
input_data_folder = 'Combined_CSV/'

# Store the column names that will be added
columns_to_add = ['Vx', 'Vy', 'Vz', 'Va', 'Vb', 'Vg']

# Read in uncompressed data and return
def read_in_data(cur):
    cur.execute('SELECT * FROM ProbeOriginal')
    return cur.fetchall()


# Delete input data collected after the last action is completed
# Also return the index of the last input
def clean_up_data(input_df, end_time):
    i = 0
    for time_stamp in input_df['Time']:
        #print('time_stamp:{} '.format(time_stamp) + ' - action_time:{} '.format(end_time))
        if abs(time_stamp - end_time) < 0.001:
           return input_df[:i + 1], i
        i += 1


# Calculates linear or angular velocity for given values
def calculate_velocity(x1,x2,t1,t2):
    return (x2 - x1) / (t2 - t1)


# Calculate velocity for a list of consecutive values
def cal_vel_for_range(pos_ori_values, time_values):
    # Check length of lists
    if len(pos_ori_values) == len(time_values):
        # initialize local variables
        velocity = []
        val = 0
        # calculate velocity and add to list
        while val < len(pos_ori_values) - 1:
            #print(pos_ori_values[val])
            #print(pos_ori_values[val+1])
            velocity.append(calculate_velocity(pos_ori_values[val],
                                               pos_ori_values[val+1],
                                               time_values[val],
                                               time_values[val+1]))
            val += 1
        return velocity


def cal_vel_for_df(input_df):
    column_index = 0
    current_columns = input_df.columns
    # loop through each column to calculate velocities
    while column_index < len(current_columns)-1:
        velocity_list = cal_vel_for_range(input_df[current_columns[column_index]], input_df['Time'])
        velocity_list.insert(0, 0)
        input_df.insert(len(input_df.columns), columns_to_add[column_index], velocity_list)
        column_index += 1
    return input_df

# adds labels to the data
def add_classes(input_df, end_time):
    action_status_list_0 = []
    i = 0
    local_index = 0
    for time_stamp in input_df['Time']:
        # print('time_stamp:{} '.format(time_stamp) + ' - action_time:{} '.format(end_time))
        class_active = 1
        if time_stamp - end_time > 0.001:
            class_active = 0
            # Create separation of dataframe
            if local_index == 0:
                local_index = i
        action_status_list_0.append(class_active)
        i += 1
    input_df.insert(len(input_df.columns), 'Square', action_status_list_0)
    action_status_list_0 = [abs(x-1) for x in action_status_list_0]
    input_df.insert(len(input_df.columns), 'Circle', action_status_list_0)
    return input_df, local_index


# fetch all databases
db_names_list = [f for f in os.listdir(source_path + database_folder) if fnmatch.fnmatch(f, '*.db')]
# fetch the file containing action times
action_time_df = pd.read_csv(source_path + action_time_csv)

#print(action_time_df.iloc[:,2])

#print(db_names_list)
#dump all data in list
counter = 0
for db_file in db_names_list:
    # Connect to each database
    db = sqlite3.connect(source_path + database_folder + db_file)
    cur = db.cursor()
    # Import data into a dataframe
    original_data = pd.DataFrame(read_in_data(cur), columns=['SrNo','X','Y','Z','A','B','G','Time'] )
    # Delete the index column
    original_data.drop('SrNo', axis=1, inplace=True)
    # Get the last time-stamp for the current performance data
    last_timestamp = action_time_df.iloc[counter,2]
    # Delete all the data recorded after the task completion
    cropped_data, index_last = clean_up_data(original_data,end_time=last_timestamp)
    # Calculate Linear and Angular velocities and add to dataframe
    input_data = cal_vel_for_df(cropped_data)
    # Rearrange the columns
    input_data = input_data[['X','Y','Z','A','B','G','Vx','Vy','Vz','Va','Vb','Vg','Time']]
    # Add one-hot-vector class columns
    complete_data, index_split = add_classes(input_data,end_time=action_time_df.iloc[counter,1])
    # Split dataframe to create separate files per action
    split_df_1 = complete_data.iloc[:index_split]
    split_df_2 = complete_data.iloc[index_split:index_last+1,:]
    # Save it as CSV file
    #complete_data.to_csv(source_path + complete_data_folder + db_file[:-2] + 'csv')
    #split_df_1.to_csv(source_path + square_data_folder + db_file[:-2] + 'csv')
    #split_df_2.to_csv(source_path + circle_data_folder + db_file[:-2] + 'csv')
    # Save as CSV in combined folder
    split_df_1.to_csv(source_path + input_data_folder + db_file[:-3] + '_S' '.csv')
    split_df_2.to_csv(source_path + input_data_folder + db_file[:-3] + '_C' '.csv')
    counter += 1




print(input_data.head())


# # Split data into features and labels
#
# # Split data into test and train
#
# # Specify hyper parameters
# epochs = 10
# n_classes = 2 #number of outputs for classification
# n_units = 100 #number of lstm cells
# n_features = 13
# batch_size = 2
#
# # Define placeholder for the data
# x_placeholder = tf.placeholder('float', [None,100,n_features])
# y_placeholder = tf.placeholder('float', [None,n_classes])


# table_names = ("ProbeOriginal", "InputData")
# # input table for the ML model
# sql_create_table = """CREATE TABLE IF NOT EXISTS InputData(
#                                 ObsNo INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
#                                 PosX FLOAT,
#                                 PosY FLOAT,
#                                 PosZ FLOAT,
#                                 OriX FLOAT,
#                                 OriY FLOAT,
#                                 OriZ FLOAT,
#                                 Vx  FLOAT,
#                                 Vy  FLOAT,
#                                 Vz  FLOAT,
#                                 Wx  FLOAT,
#                                 Wy  FLOAT,
#                                 Wz  FLOAT,
#                                 TimeElapsed FLOAT"""
# def create_table(db_file, sql_table):
#     db_conn = sqlite3.connect(source_path + db_file)
#     db_cursor = db_conn.cursor()
#     try:
#         db_cursor.execute(sql_create_table)
#     except Error as e:
#         print(e)
#     db_cursor.close()
#     db_conn.close()


# def insert_into_table(db_file, table_names):
#     db_conn = sqlite3.connect(source_path + db_file)
#     db_cursor = db_conn.cursor()
#     db_cursor.execute("""INSERT INTO InputData(PosX, PosY, PosZ,
#                                              OriX, OriY, OriZ, TimeElapsed)
#                                 VALUES(?,?,?,?,?,?)""", entities)
# entities = None
