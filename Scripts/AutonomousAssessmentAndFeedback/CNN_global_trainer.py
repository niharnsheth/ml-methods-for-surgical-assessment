## --- Libraries   ---  ##
# File imports and aggregates data from multiple databases
import os
import fnmatch
import pandas as pd
import numpy as np
import random

import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import GlobalAvgPool1D, Flatten
# from tensorflow.keras.regularizers import l2

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
save_model = '/08092021_refCNN_w180'


##  ---  Define hyper parameters  ---  #
skill_levels = 3

no_sensors = 1
features_per_sensor = 13
num_of_labels = 3
n_features = no_sensors * features_per_sensor
n_classes = 3  # number of outputs for classification

# ------------- ideal training values  ---------------- #
# epochs = [[50, 50], [50, 50, 75, 50]]
# sliding_window = [[100, 175], [100, 80, 175, 150]]
# batch_size = [[75, 125], [150, 75, 100, 125]]
# ----------------------------------------------------#

epochs = [[100, 100], [100, 100, 100, 100]]
sliding_window = [[180, 180], [180, 180, 180, 180]]
window_step_size = 1
batch_size = [[100, 100], [100, 100, 100, 100]]
learning_rate = 0.001
set_rand_seed = 11
random.seed(set_rand_seed)


##  ---   Make motion windows   --- #
# return an input and output data frame with motion windows
def create_motion_windows(window_span, df_to_change, step_size, number_of_features, number_of_labels):
    local_feature_df = []
    local_label_df = []
    # steps = range(len(df_to_change) - window_span)
    time_index = 0
    while time_index + window_span < len(df_to_change):
        feat_local = df_to_change.iloc[time_index:time_index + window_span, :-number_of_labels].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        lab_loc = df_to_change.iloc[time_index + window_span, number_of_features:].reset_index(drop=True).to_numpy()
        local_feature_df.append(feat_local)
        local_label_df.append(lab_loc)
        time_index += step_size
    return local_feature_df, local_label_df


# Return index for annotation
def check_experience_level(experience):
    if fnmatch.fnmatch(experience, 'Novice'):
        return 0
    elif fnmatch.fnmatch(experience, 'Intermediate'):
        return 1
    elif fnmatch.fnmatch(experience, 'Expert'):
        return 2
    else:
        return 3


# ## ----------------- FINALIZED  CNN 1D ----------------------------------------- #
#
# # --- tf.Keras implementation of LSTM layers --- #
# model = Sequential()
# model.add(Conv1D(filters=38, kernel_size=2, activation='relu', input_shape=(None, n_features)))
# model.add(MaxPool1D(pool_size=2, strides=2))
# model.add(Dropout(0.25))
# model.add(Conv1D(filters=76, kernel_size=2, activation='relu'))
# model.add(MaxPool1D(pool_size=2, strides=2))
# model.add(Dropout(0.25))
# model.add(Conv1D(filters=152, kernel_size=2, activation='relu'))
# model.add(MaxPool1D(pool_size=2, strides=2))
# model.add(Dropout(0.25))
# model.add(Conv1D(filters=304, kernel_size=2, activation='relu'))
# model.add(MaxPool1D(pool_size=2, strides=2))
# model.add(Dropout(0.25))
# # model.add(Flatten())
# model.add(GlobalAvgPool1D())
# model.add(Dropout(0.5))
# model.add(Dense(152, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(38, activation='relu'))
# model.add(Dense(n_classes, activation='softmax'))
# opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-3)
#
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#

## ----------------- REFERENCE  CNN 1D ----------------------------------------- #
# --- tf.Keras implementation of LSTM layers --- #
model = Sequential()
model.add(Conv1D(filters=38, kernel_size=2, activation='relu', input_shape=(180, n_features)))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=76, kernel_size=2, activation='relu'))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=152, kernel_size=2, activation='relu'))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-3)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


## --- Training process  ---  #
print(' ------------- Training   ------------')

# Create folder to save the model and results
os.mkdir(source_path + save_to_folder + save_model)
# Create folder to save all the plots
os.mkdir(source_path + save_to_folder + save_model + '/' + 'Graphs')
# train for all surgical procedures
for surgery_selected in range(0, len(surgery_name_list)):
    manually_annotated_labels = pd.read_csv(source_path + "/" + surgery_name_list[surgery_selected][1:] + ".csv")

    # train for each surgical task with procedure
    surgical_tasks = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')
    for action_selected in range(len(surgical_tasks)):
        print("Surgery: " + surgery_name_list[surgery_selected] + " -  Action: " + surgical_tasks[action_selected])
        # get data of surgical tasks
        csv_list = [f for f in os.listdir(source_path + input_folder +
                                          surgery_name_list[surgery_selected] + '/' +
                                          surgical_tasks[action_selected] + '/')
                    if fnmatch.fnmatch(f, '*.csv')]

        # initialize input and output list
        feature_list_main = [[] for _ in range(3)]
        label_list_main = [[] for _ in range(3)]

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

            # Get experience level
            file_idx = manually_annotated_labels.index[manually_annotated_labels['PerformanceName'] == file[:-4]]
            experience_level = manually_annotated_labels.iloc[file_idx][surgical_tasks[action_selected]].iloc[0]
            exp_index = check_experience_level(experience_level)
            # # Get experience level
            # split_list = file.split('_')
            # experience_level = split_list[1]
            # exp_index = check_experience_level(experience_level)

            # create motion windows and separate data into input and output
            # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)
            feature_list, label_list = create_motion_windows(sliding_window[surgery_selected][action_selected], df,
                                                             window_step_size,
                                                             features_per_sensor,
                                                             num_of_labels)

            # create list of windows
            feature_list_main[exp_index].extend(feature_list)
            label_list_main[exp_index].extend(label_list)
        # count number of windows for each class
        a = []
        for i in range(len(feature_list_main)):
            a.append(len(feature_list_main[i]))
            print("Number of windows: " + str(len(feature_list_main[i])))
        min_window = min(a)

        # randomly select min_windows from each class
        print("Length of feature list: " + str(len(feature_list_main)))
        for i in range(len(feature_list_main)):
            if min_window != len(feature_list_main[i]):
                feature_list_main[i] = random.choices(feature_list_main[i], k=min_window)
                label_list_main[i] = label_list_main[i][0:min_window]

        # combine all lists to one
        input_feature_list = []
        output_label_list = []
        input_feature_list.extend(feature_list_main[0])
        input_feature_list.extend(feature_list_main[1])
        input_feature_list.extend(feature_list_main[2])
        output_label_list.extend(label_list_main[0])
        output_label_list.extend(label_list_main[1])
        output_label_list.extend(label_list_main[2])
        feature_list_main.clear()
        label_list_main.clear()

        # shuffle data before training model
        combined_list = list(zip(input_feature_list, output_label_list))
        random.shuffle(combined_list)
        input_feature_list, output_label_list = zip(*combined_list)
        combined_list.clear()

        # reshape to train
        input_feature_list = np.reshape(input_feature_list,
                                        (len(input_feature_list),
                                         sliding_window[surgery_selected][action_selected],
                                         n_features))
        input_feature_list = np.array(input_feature_list)
        output_label_list = np.array(output_label_list)
        # get total number of batches
        total_motion_n_windows = len(input_feature_list)
        print("Total no. of motion windows for: " + surgery_name_list[surgery_selected][1:] +
              '- ' + surgical_tasks[action_selected] + ': ' + str(total_motion_n_windows))

        n_batches = int(total_motion_n_windows / batch_size[surgery_selected][action_selected])
        print("Total no. of batches for: " + surgery_name_list[surgery_selected][1:] +
              '- ' + surgical_tasks[action_selected] + ': ' + str(n_batches))

        # split data for training and testing
        x_train, x_test, y_train, y_test = train_test_split(input_feature_list, output_label_list, test_size=0.15, random_state=set_rand_seed)
        # train
        history = model.fit(x_train, y_train,
                            epochs=epochs[surgery_selected][action_selected],
                            batch_size=batch_size[surgery_selected][action_selected],
                            validation_split=0.15,
                            verbose=2)
        # display summary of training
        model.summary()

        # plot losses
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Model loss - ' + surgery_name_list[surgery_selected] +
                  ' - ' + action_name_list[surgery_selected][action_selected])
        plt.ylabel('loss value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected][1:] + '_' + str(action_selected) + '_' + 'loss' + '.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # plot accuracies
        plt.plot(history.history['accuracy'], label='acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Model accuracy' + surgery_name_list[surgery_selected] +
                  ' - ' + action_name_list[surgery_selected][action_selected])
        plt.ylabel('accuracy value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected][1:] + '_' + str(action_selected) + '_' + 'acc' + '.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # test the model
        model.evaluate(x_test, y_test, batch_size=n_batches, verbose=2)

        # Create folder to save trained model
        os.mkdir(source_path + save_to_folder + save_model + '/' +
                 surgery_name_list[surgery_selected] + '_' + str(action_selected))
        # save the trained model
        model.save(source_path + save_to_folder + save_model +
                   '/' + surgery_name_list[surgery_selected] +
                   '_' + str(action_selected) + '/', save_format='tf')

        # save the history of training to csv file
        train_hist_df = pd.DataFrame(history.history)
        train_hist_df.to_csv(source_path + save_to_folder + save_model + '/'
                             + surgery_name_list[surgery_selected] + '_'
                             + str(action_selected) + '/'
                             + action_name_list[surgery_selected][action_selected] + '.csv', index=False)
