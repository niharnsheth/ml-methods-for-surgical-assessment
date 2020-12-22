import os
import pandas

# path to the folder containing the files
root_dir = os.getcwd()+'/../'+'CSVDataset1/'
directory_files = os.listdir(root_dir)

# Read in the file data into a dataframe
#df = pandas.read_csv(root_dir + directory_files[0])
# Delete the index column
#df.drop(['r'], axis=1, inplace=True)

# store the the column headers
#current_columns = df.columns
#print(len(current_columns))
#print(current_columns[0])

# Store the column names that will be added
columns_to_add = ['Vx', 'Vy', 'Vz', 'Va', 'Vb', 'Vg']
#print(columns_to_add[0])


# Calculates linear and angular velocity for given values
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


# Calculate velocity for multiple lists in a cvs file
def cal_vel_for_file(path_to_file, name_of_file):
    # Read in the file data into a dataframe
    df_local = pandas.read_csv(path_to_file+name_of_file)
    # Check if time column exits
    if 'T' in df_local.columns:
        # Delete the index column
        df_local.drop('r', axis=1, inplace=True)
        # store the the column headers
        current_columns = df_local.columns
        print(current_columns)
        print("Number of colums are: {}".format(len(current_columns)))
        #print(current_columns)
        column_index = 0
        # Loop through each column through consecutive values to calculate velocity
        while column_index < len(current_columns) - 1:
            print("column_id = {}".format(column_index))
            velocity_list = cal_vel_for_range(df_local[current_columns[column_index]], df_local['T'])
            velocity_list.insert(0, 0)
            df_local.insert(len(df_local.columns), columns_to_add[column_index], velocity_list)
            column_index += 1
    return df_local

#velocity_list = cal_vel_for_range(df['X'], df['T'])
#velocity_list.insert(0,0)
#df.insert(len(df.columns),'Vx',velocity_list)
#df['Vx'] = velocity_list
#print(velocity_list)

for dataset in directory_files:
    print(dataset)
    df = cal_vel_for_file(root_dir, dataset)
    #print("Creting a new file now")
    df.to_csv(os.getcwd()+r'/../' + r'CSVModifiedData1/' + dataset)
    #print(df.columns)

