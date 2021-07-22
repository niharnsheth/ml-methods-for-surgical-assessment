## --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import csv
from tensorflow.python import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

models_list = ['CNN', 'LSTM', 'Siamese']
test_path = 0 # for test_data_path
surgery_selected = 1
model_selected = 0 # for save_to_folder, saved_model, test_save_to_path
# task_model_selected = 2

#sliding_window = [[150, 200], [125, 100, 200, 150]]
sliding_window = [[100, 150], [125, 100, 200, 150]]

num_of_labels = 3
num_of_features = 13
n_features = 13

# File path to the database files
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
# source_path = os.getcwd() + "/../../../../Nihar/ML-data/SurgicalData"
source_path = os.getcwd() + '/../../../../Nihar/ML-data/SurgicalData/Manually_Cleaned_And_Annotated_06272021'
# test_data_path = ['/TestData/Annotated']
test_data_path = ['/TestData/TestingDataForClassification']
surgery_name_list = ['/Pericardiocentesis', '/Thoracentesis']


# saved_to_folder = ['/Results/1D_CNN/AutoAnnotated', '/Results/LSTM/AutoAnnotated']
saved_to_folder = ['/Results/1D_CNN', '/Results/LSTM']
# saved_model = ['/06112021_1', '/06092021', '/06202021']
saved_model = ['/07112021', '/06222021']

test_save_to_path = ['/TestData/Results/1D_CNN/AutoAnnotated/06222021',
                     '/TestData/Results/LSTM/AutoAnnotated/06222021']

models_per_task = ['Pericardiocentesis_0', 'Pericardiocentesis_2',
                   'Thoracentesis_0', 'Thoracentesis_1', 'Thoracentesis_2', 'Thoracentesis_3']


##  ---   Make motion windows   --- #
# return an input and output data frame with motion windows
def create_motion_windows(window_span, df_to_change, step_size, number_of_features, number_of_labels):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - window_span)
    time_index = 0
    while time_index + window_span < len(df_to_change):
        a = df_to_change.iloc[time_index:time_index + window_span, :-number_of_labels].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[time_index + window_span, number_of_features:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
        time_index += step_size
    return local_feature_df, local_label_df

# Create folder to save all the plots
#os.mkdir(source_path + test_save_to_path[model_selected] + '/' + 'Graphs')


## ---   Test classification models   --- #

surgical_tasks_list = os.listdir(source_path + test_data_path[test_path] + surgery_name_list[surgery_selected] + '/')
# get all the models saved for each task
for surgical_task in range(len(surgical_tasks_list)):
    # find the saved model to use
    model_for_task = surgery_name_list[surgery_selected] + '_' + surgical_tasks_list[surgical_task]
    loaded_model = load_model(source_path + saved_to_folder[model_selected] +
                              saved_model[model_selected] + '/' + model_for_task,
                              custom_objects=None, compile=True)
    csv_list = [f for f in os.listdir(source_path + test_data_path[test_path] +
                                      surgery_name_list[surgery_selected] + '/' +
                                      surgical_tasks_list[surgical_task] + '/')
                if fnmatch.fnmatch(f, '*.csv')]
    # initialize list to store predictions
    predictions = []
    # feature_list, label_list = []

    # train for surgical task
    for file in csv_list:
        df = pd.read_csv(source_path + test_data_path[test_path] +
                         surgery_name_list[surgery_selected] + '/' +
                         surgical_tasks_list[surgical_task] + '/' + file)

        # create motion windows and separate data into input and output
        # use the same step size as the size of the window
        window_size = sliding_window[surgery_selected][surgical_task]
        feature_list, label_list = create_motion_windows(window_size,
                                                         df,
                                                         window_size,
                                                         num_of_features,
                                                         num_of_labels)
        # create list of windows
        feature_list = np.reshape(feature_list, (len(feature_list),
                                                 sliding_window[surgery_selected][surgical_task],
                                                 n_features))
        # label_list = np.array(label_list)

        prediction = loaded_model.predict(feature_list)

        print(" --------    Current Surgical Task: " + surgical_tasks_list[surgical_task] + "      ---------")
        print(" --------    Predicting for file: " + file + "      ---------")
        print(" --------    Using model: " + model_for_task + "      ---------")
        print("   Novice       Intermediate       Expert")
        # read the file into a data-frame
        print(prediction * 100)
        print("----------------------------------------------------------------------------")
        # plot losses
        # x_axis_val = np.arange(len(prediction))
        #
        # plt.bar(x_axis_val - 0.2, prediction[0][0], 0.3, label='Novice')
        # plt.bar(x_axis_val, prediction[0][1], 0.3, label='Intermediate')
        # plt.bar(x_axis_val + 0.2, prediction[0][2], 0.3, label='Expert')
        # # plt.plot(prediction, label='predictions')
        # # plt.plot(label_list, label='true_value')
        # plt.title('Prediction: ' + surgery_name_list[surgery_selected] + ' - ' +
        #           str(surgical_task) + ' ' + file)
        # plt.ylabel('Predictions')
        # plt.xlabel('Windows')
        # plt.legend(loc="upper left")
        # plt.savefig(source_path + test_save_to_path[model_selected] + '/' + 'Graphs' + '/' +
        #             surgery_name_list[surgery_selected][1:] + '_' + str(surgical_task) + '_' + 'predictions' + '.png',
        #             dpi=300, bbox_inches='tight')
        # plt.show()


## testing
#
# max_predicted_val = np.amax(prediction[[0]])
# print("max value: " + str(max_predicted_val))
# print(len(prediction))
# print(prediction[0][0])
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData/Manually_Cleaned_And_Annotated_06272021/Results/1D_CNN/07142021'


names_of_models = [['Pericardiocentesis_0', 'Pericardiocentesis_1'],
                   ['Thoracentesis_0', 'Thoracentesis_1',
                    'Thoracentesis_2', 'Thoracentesis_3' ]]
models_list = []
for surg_proc in range(len(names_of_models)):
    loc_model_list = []
    for mods in range(len(names_of_models[surg_proc])):
        loc_model_list.append(load_model(source_path + '/' + names_of_models[surg_proc][mods] + '/',
                                         custom_objects=None,
                                         compile=True))
    models_list.append(loc_model_list)