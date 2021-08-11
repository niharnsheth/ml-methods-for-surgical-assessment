## --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import pandas as pd
import numpy as np
import random

import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

# File path to the database files
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
# source_path = os.getcwd() + '/../../../../Nihar/ML-data/SurgicalData'
source_path = os.getcwd() + '/../../../../Nihar/ML-data/SurgicalData/ManuallyCleaned_06262021'
#source_path = os.getcwd() + "/../../../../SurgicalData"
surgery_selected = 1
#action_selected = 2

surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

input_folder = '/AutoAnnotated'
save_model = '/07082021'
save_to_folder = '/Results/LSTM/AutoAnnotated'


## 1 ---  Define hyper parameters  ---  #
skill_levels = 3

no_sensors = 1
features_per_sensor = 13
n_features = no_sensors * features_per_sensor
n_classes = 3  # number of outputs for classification
epochs = [[35, 35], [40, 35, 35, 35]]

sliding_window = [[100, 225], [100, 80, 200, 150]]
window_step_size = 1

batch_size = [[200, 125], [150, 75, 100, 125]]
learning_rate = 0.001
set_rand_seed = 7
random.seed(set_rand_seed)

n_units = 250  # number of lstm cells


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


# Return index for annotation
def check_experience_level(experience):
    if fnmatch.fnmatch(experience, 'Resident'):
        return 0
    elif fnmatch.fnmatch(experience, 'Fellow'):
        return 1
    else:
        return 2


# Return index for annotation
def check_experience_level_1(experience):
    if fnmatch.fnmatch(experience, 'Novice'):
        return 0
    elif fnmatch.fnmatch(experience, 'Intermediate'):
        return 1
    elif fnmatch.fnmatch(experience, 'Expert'):
        return 2
    else:
        return 3

# -----------------   LSTM ----------------------------------------- #
## --- tf.Keras implementation of LSTM layers --- #
model = Sequential()
opt = tf.keras.optimizers.Adam(lr=learning_rate, momentum=0.9, decay=1e-3)
#model.add(LSTM(n_units, input_shape=(sliding_window, n_features),return_sequences=True, name='lstm_layer_1'))
#model.add(LSTM(n_units, name='lstm_layer_2'))
model.add(LSTM(n_units, input_shape=(None, n_features), name='lstm_layer_1'))
model.add(Dropout(0.5))
model.add(LSTM(int(n_units/2), name='lstm_layer_2'))
model.add(Dropout(0.5))
model.add(Dense(n_units, activation='relu', name='projection_layer'))
model.add(Dense(n_classes, activation='softmax', name='output_layer'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


## --- Training process  ---  #
print(' ------------- Training   ------------')
total_runs = 5
# Create folder to save all the plots
os.mkdir(source_path + save_to_folder + save_model + '/' + 'Graphs')

# train for all surgical procedures
for surgery_selected in range(0, len(surgery_name_list)):
    # train for each surgical task with procedure
    surgical_tasks = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')
    # get the file that containes the manually annotated label details
    manually_annotated_labels = pd.read_csv(source_path + "/" + surgery_name_list[surgery_selected][1:] + ".csv")

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
            # remove null value rows from sample
            check_if_null = df.isnull().values.any()
            print(file + " has null values: " + str(check_if_null))
            df = df.dropna(how='any', axis=0)

            # # Get experience level
            # split_list = file.split('_')
            # experience_level = split_list[1]
            # exp_index = check_experience_level(experience_level)
            # Get experience level
            file_idx = manually_annotated_labels.index[manually_annotated_labels['PerformanceName'] == file[:-4]]
            experience_level = manually_annotated_labels.iloc[file_idx][surgical_tasks[action_selected]].iloc[0]
            exp_index = check_experience_level_1(experience_level)


            # create motion windows and separate data into input and output
            # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)w
            feature_list, label_list = create_motion_windows(sliding_window[surgery_selected][action_selected], df,
                                                             window_step_size,
                                                             features_per_sensor,
                                                             n_classes)
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
        output_label_list = np.array(output_label_list)
        # get total number of batches
        total_motion_n_windows = len(input_feature_list)
        print("Total no. of motion windows for: " + surgery_name_list[surgery_selected][1:] +
              '- ' + surgical_tasks[action_selected] + ': ' + str(total_motion_n_windows))

        n_batches = int(total_motion_n_windows / batch_size[surgery_selected][action_selected])
        print("Total no. of batches for: " + surgery_name_list[surgery_selected][1:] +
              '- ' + surgical_tasks[action_selected] + ': ' + str(n_batches))

        # split data for training and testing
        x_train, x_test, y_train, y_test = train_test_split(input_feature_list,
                                                            output_label_list,
                                                            test_size=0.15,
                                                            random_state=set_rand_seed)
        # train
        history = model.fit(x_train, y_train,
                            epochs=epochs[surgery_selected][action_selected],
                            batch_size=batch_size[surgery_selected][action_selected],
                            validation_split=0.2,
                            verbose=2)
        # display summary of training
        model.summary()

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Model loss - ' + surgery_name_list[surgery_selected] + ' - ' + surgical_tasks[action_selected])
        plt.ylabel('loss value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected] + '_' + surgical_tasks[action_selected] + '_' + 'loss' + '.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        plt.plot(history.history['accuracy'], label='acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Model accuracy - ' + surgery_name_list[surgery_selected] + ' - ' + surgical_tasks[action_selected])
        plt.ylabel('accuracy value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected] + '_' + surgical_tasks[action_selected] + '_' + 'acc' + '.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # test the model
        model.evaluate(x_test, y_test, batch_size=10, verbose=2)

        # Create folder to save trained model
        os.mkdir(source_path + save_to_folder + save_model + '/' +
                 surgery_name_list[surgery_selected] + '_' + surgical_tasks[action_selected])
        # save the trained model
        model.save(source_path + save_to_folder + save_model +
                   '/' + surgery_name_list[surgery_selected] +
                   '_' + surgical_tasks[action_selected] + '/', save_format='tf')
