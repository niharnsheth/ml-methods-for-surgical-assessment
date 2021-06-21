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
surgery_selected = 1
test_path = 0
model_selected = 0
task_model_selected = 2

sliding_window = 100
step_size = sliding_window
n_features = 13

# File path to the database files
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
#source_path = os.getcwd() + "/../../../../Nihar/ML-data/SurgicalData"
test_data_path = ['/TestData/Annotated', '/TestData/AnnotatedForSiamese']
surgery_name_list = ['/Pericardiocentesis', '/Thoracentesis']

saved_to_folder = ['/Results/1D_CNN/AutoAnnotated', '/Results/LSTM/AutoAnnotated', '/Results/Siamese/AutoAnnotated']
saved_model = ['/06112021_1', '/06092021', '/06202021']

models_per_task = ['Pericardiocentesis_0', 'Pericardiocentesis_2',
                   'Thoracentesis_0', 'Thoracentesis_1', 'Thoracentesis_2', 'Thoracentesis_3']

##  ---   Make motion windows   --- #

# return an input and output data frame with motion windows
def create_motion_windows(window_span, df_to_change, step_size, num_of_labels):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - window_span)
    time_index = 0
    while time_index + window_span < len(df_to_change):
        a = df_to_change.iloc[time_index:time_index + window_span, :-num_of_labels].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[time_index + window_span, 13:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
        time_index += step_size
    return local_feature_df, local_label_df


surgical_tasks_list = os.listdir(source_path + test_data_path[test_path] + surgery_name_list[surgery_selected] + '/')
# get all the models saved for each task
#models_per_task = os.listdir(source_path + saved_to_folder[model_selected] + saved_model[model_selected])
##
for surgical_task in surgical_tasks_list:
    # find the saved model to use
    model_for_task = surgery_name_list[surgery_selected] + '_' + surgical_task
    loaded_model = load_model(source_path + saved_to_folder[model_selected] +
                              saved_model[model_selected] + '/' + model_for_task,
                              custom_objects=None, compile=True)
    csv_list = [f for f in os.listdir(source_path + test_data_path[test_path] +
                                      surgery_name_list[surgery_selected] + '/' +
                                      str(surgical_task) + '/')
                if fnmatch.fnmatch(f, '*.csv')]
    predictions = []
    # train for surgical task
    for file in csv_list:
        print(" --------    Current Surgical Task: " + surgical_task + "      ---------")
        print(" --------    Predicting for file: " + file + "      ---------")

        # read the file into a data-frame
        df = pd.read_csv(source_path + test_data_path[test_path] +
                         surgery_name_list[surgery_selected] + '/' +
                         str(surgical_task) + '/' + file)

        # create motion windows and separate data into input and output
        # use the same step size as the size of the window
        feature_list, label_list = create_motion_windows(sliding_window, df, step_size, 3)
        # create list of windows
        feature_list = np.reshape(feature_list, (len(feature_list), sliding_window, n_features))
        label_list = np.array(label_list)

        prediction = loaded_model.predict(feature_list)
        print(prediction * 100)
        print("----------------------------------------------------------------------------")

        # for window in range(len(feature_list)):
        #     prediction = loaded_model.predict(feature_list[window])
        #     print(predictions)
        #     predictions.append(prediction)
        #
        # plt.plot(history.history['loss'], label='loss')
        # plt.plot(history.history['val_loss'], label='val_loss')
        # plt.title('Model loss - ' + surgery_name_list[surgery_selected] + ' - ' + str(action_selected))
        # plt.ylabel('loss value')
        # plt.xlabel('epoch')
        # plt.legend(loc="upper left")
        # plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
        #             surgery_name_list[surgery_selected] + '_' + str(action_selected) + '_' + 'loss' + '.png',
        #             dpi=300, bbox_inches='tight')
## testing

max_predicted_val = np.amax(prediction[[0]])
print("max value: " + str(max_predicted_val))