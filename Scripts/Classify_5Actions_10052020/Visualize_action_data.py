
# Script visualizes data of surgical actions
## Import library and data
import os
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# ---- File path to the database files  ---- #

# source folder, change based on RUN options (entire script / parts of scripts)
source_path = os.getcwd()  + '/Data/'
#source_path = os.getcwd()  + '/../..' +  '/Data/'
# file paths to orginal data folders
original_data_folder = ['5 Actions_10032020/OriginalData/ChloraPrep/',
                        '5 Actions_10032020/OriginalData/NeedleInsertion/',
                        '5 Actions_10032020/OriginalData/Dilator/',
                        '5 Actions_10032020/OriginalData/Cut/',
                        '5 Actions_10032020/OriginalData/Tracing/']
# file paths to normalized and modified data folders
processed_data_folder = ['5 Actions_10032020/ChloraPrep/',
                         '5 Actions_10032020/NeedleInsertion/',
                         '5 Actions_10032020/Dilator/',
                         '5 Actions_10032020/Cut/',
                         '5 Actions_10032020/Tracing/']

prepared_data_folder = '5 Actions_10032020/PreparedData/'

# Select single folder for testing - IGNORE
data_folder_ex =  '5 Actions_10032020/OriginalData/Tracing/'

save_to_folder = '4sensors_6actions_dataset/Prepared Data/'
# obtain list of files
#csv_list_1 = [f for f in os.listdir(source_path) if fnmatch.fnmatch(f, '*.csv')]
csv_list_1 = [f for f in os.listdir(source_path + data_folder_ex) if fnmatch.fnmatch(f, '*.csv')]

## Plot data to find range of normalization
# obtain list of files

labels_legend = ['ChloraPrep', 'NeedleInsert', 'Dilator', 'Cut', 'Tracing']
action_legends = [0,0,0]
fig, ax1 = plt.subplots()

for i in range(len(processed_data_folder)):
    csv_list_1 = [f for f in os.listdir(source_path + processed_data_folder[i]) if fnmatch.fnmatch(f, '*.csv')]
    main_df = pd.DataFrame()
    seq_df = pd.DataFrame()
    for file in csv_list_1:
        # df = pd.read_csv(source_path + original_data_folder[i] + file)
        df = pd.read_csv(source_path + processed_data_folder[i] + file)
        length_of_df = df.shape[0]
        seq_df = seq_df.append(pd.DataFrame(range(0,length_of_df)), ignore_index=True)
        main_df = main_df.append(df,ignore_index=True)

    main_np_arr = main_df.to_numpy()
    seq_np_arr = seq_df.to_numpy()
    # ax1.scatter(main_np_arr[:,8],main_np_arr[:,4], label=labels_legend[i])
    ax1.scatter(main_np_arr[:,13],seq_np_arr[:,0], label=labels_legend[i])

plt.legend()
plt.xlim([-200,200])
plt.xticks(np.arange(-200,200,25))
plt.title('Z angular Velocity')
plt.xlabel('vel_values')
plt.ylabel('time_steps')
plt.show()


## Plot STD and mean
main_df = pd.DataFrame()
list_of_std =[]
mean_np_arr = np.empty([len(csv_list_1),6])
std_np_arr = np.empty([len(csv_list_1),6])
max_np_arr = np.empty([len(csv_list_1),6])
min_np_arr = np.empty([len(csv_list_1),6])

file_index = 0
# traverse and find standard deviation, mean of all samples to plot
for file in csv_list_1:
    # read sample data
    df = pd.read_csv(source_path + data_folder_ex + file)
    # drop time and serial columns
    df = df.drop(['SId','PT','OT'], axis=1)
    # convert to numpy array
    local_numpy = df.to_numpy()
    # get std for each column
    for col in range(local_numpy.shape[1]):
        #list_of_std.append(np.std(local_numpy[:,0]))
        std_np_arr[file_index,col] = np.std(local_numpy[:,col])
        mean_np_arr[file_index,col] = np.mean(local_numpy[:,col])
        max_np_arr[file_index,col] = np.max(local_numpy[:,col])
        min_np_arr[file_index,col] = np.min(local_numpy[:,col])
    file_index += 1


    #main_df = main_df.append(df, ignore_index=True)

max_df = pd.DataFrame(data= max_np_arr)
min_df = pd.DataFrame(data=min_np_arr)
print(max_df.describe())
print(max_df.info())
print(min_df.describe())
print(min_df.info())
# print(df.describe())
# desc = df.describe()
# print(df.info())
# info = df.info()

#fig, ax = plt.subplot()
x_line, = plt.plot(range(0,35),std_np_arr[:,3], label='X')
y_line, = plt.plot(range(0,35),std_np_arr[:,4], label='Y')
z_line, = plt.plot(range(0,35),std_np_arr[:,5], label='Z')

plt.legend(handles=[x_line,y_line,z_line])
plt.title('ChloraPrep: STD of Angle value')
plt.xlabel('Sample No.')
plt.ylabel('STD')
plt.show()

## 3D plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# numpy_arr = main_df.to_numpy()
# x = numpy_arr[:,1]
# y = numpy_arr[:,2]
# z = numpy_arr[:,3]

# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
# ax.scatter3D(x,y,z , c=z, cmap='Greens')
# plt.show()



## Plot normalized data
action_no = 1
file_no = 2

original_file_list = [f for f in os.listdir(source_path + original_data_folder[action_no]) if fnmatch.fnmatch(f, '*.csv')]
processed_file_list = [f for f in os.listdir(source_path + processed_data_folder[action_no]) if fnmatch.fnmatch(f, '*.csv')]

ori_data_df = pd.read_csv(source_path +
                          original_data_folder[action_no] +
                          original_file_list[file_no])
proc_data_df = pd.read_csv(source_path +
                           processed_data_folder[action_no] +
                           processed_file_list[file_no])

labels_legend = ['ChloraPrep', 'NeedleInsert', 'Dilator']

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax_1 = plt.axes(projection='3d')

ori_data_np = ori_data_df.to_numpy()
x = ori_data_np[:,4]
y = ori_data_np[:,5]
z = ori_data_np[:,6]

proc_data_np = proc_data_df.to_numpy()
x_1 = proc_data_np[:,4]
y_1 = proc_data_np[:,5]
z_1 = proc_data_np[:,6]


# Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
#ax.scatter3D(x, y, z, c=z, cmap='Greens')
ax.scatter3D(x_1, y_1, z_1, c=z_1, cmap='Reds')
plt.show()

#plt.show()
##

