## --- Libraries   ---  ##
# File imports and aggregates data from multiple databases
import os
import fnmatch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# File path to + the database files
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
# source_path = os.getcwd() + '/../../../../Nihar/ML-data/SurgicalData/Data_04152021'
# source_path = os.getcwd() + '/../../../../Nihar/ML-data/SurgicalData/ManuallyCleaned_06262021'
source_path = os.getcwd() + '/../../../../Nihar/ML-data/SurgicalData/Manually_Cleaned_And_Annotated_06272021'

surgery_selected = 0
# action_selected = 2

# surgery_name_list = ['/Pericardiocentesis', '/Thoracentesis']
surgery_name_list = ['/Pericardiocentesis', '/Thoracentesis']
action_name_list = [['Chloraprep', 'Needle Insertion'],
                    ['Chloraprep', 'Scalpel Incision', 'Trocar Insertion', 'Anesthetization']]
# surgery_name_list = ['/Thoracentesis']
input_folder = '/TrainingDataForClassification'
save_to_folder = '/Results/1D_CNN'
# save_model = '/08092021_myCNN_w100'

total_lengths = []
for surgery_selected in range(0, len(surgery_name_list)):
    manually_annotated_labels = pd.read_csv(source_path + "/" + surgery_name_list[surgery_selected][1:] + ".csv")

    # train for each surgical task with procedure
    surgical_tasks = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')
    total_action_lengths = []
    for action_selected in range(len(surgical_tasks)):
        print("Surgery: " + surgery_name_list[surgery_selected] + " -  Action: " + surgical_tasks[action_selected])
        # get data of surgical tasks
        csv_list = [f for f in os.listdir(source_path + input_folder +
                                          surgery_name_list[surgery_selected] + '/' +
                                          surgical_tasks[action_selected] + '/')
                    if fnmatch.fnmatch(f, '*.csv')]

        length_of_file = 0

        # train for surgical task
        for file in csv_list:
            # read the file into a data-frame
            df = pd.read_csv(source_path + input_folder +
                             surgery_name_list[surgery_selected] +
                             '/' + surgical_tasks[action_selected] +
                             '/' + file)
            check_if_null = df.isnull().values.any()
            print(file + " has null values: " + str(check_if_null))
            df = df.dropna(how='any', axis=0)
            length_of_file += len(df)

        avg_length_of_action = length_of_file/len(csv_list)
        print("Average lenght of action " + str(avg_length_of_action))
        total_action_lengths.append(avg_length_of_action)

    total_lengths.append(total_action_lengths)
total_lengths_df = pd.DataFrame(total_lengths, columns=['0', '1', '2', '3'])
total_lengths_df.to_csv(source_path + input_folder + '/' + 'AverageTaskLength.csv')