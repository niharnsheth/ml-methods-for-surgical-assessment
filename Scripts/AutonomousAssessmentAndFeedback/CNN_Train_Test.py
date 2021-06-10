## --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import pandas as pd
import numpy as np
from random import random

import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

# File path to the database files
source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
#source_path = os.getcwd() + "/../../../../SurgicalData"
surgery_selected = 1
#action_selected = 2

surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

data_folder = '5 Actions_10032020/PreparedData/'

input_folder = '/Annotated'
save_model = '/06092021'
save_to_folder = '/Results/LSTM/AutoAnnotated'


## 1 ---  Define hyper parameters  ---  #
skill_levels = 3

no_sensors = 1
features_per_sensor = 13
n_features = no_sensors * features_per_sensor
n_classes = 3 #number of outputs for classification
epochs = 15

sliding_window = 100
batch_size = 300


##  ---   Make motion windows   --- #

# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-3].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step+no_windows, 13:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df

# -----------------   LSTM ----------------------------------------- #
learning_rate = 0.001
n_units = 150 # number of lstm cells
## --- tf.Keras implementation of LSTM layers --- #
model = Sequential()
opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=1e-2/epochs)
#model.add(LSTM(n_units, input_shape=(sliding_window, n_features),return_sequences=True, name='lstm_layer_1'))
#model.add(LSTM(n_units, name='lstm_layer_2'))
#model.add(Dropout(0.5))
model.add(LSTM(n_units, input_shape=(sliding_window, n_features), name='lstm_layer_1'))
model.add(Dense(n_units/2, activation='relu', name='projeciton_layer'))
model.add(Dense(n_classes, activation='softmax', name='output_layer'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])


# -----------------   1D CNN   ----------------------------------------- #




## --- Training process  ---  #
print(' ------------- Training   ------------')
total_runs = 5
# Create folder to save all the plots
os.mkdir(source_path + save_to_folder + save_model + '/' + 'Graphs')
# train for all surgical procedures
for surgery_selected in range(0,len(surgery_name_list)):
    # train for each surgical task with procedure
    surgical_tasks = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')
    for action_selected in surgical_tasks:
        # get data of surgical tasks
        csv_list = [f for f in os.listdir(source_path + input_folder +
                                          surgery_name_list[surgery_selected] + '/' +
                                          str(action_selected) + '/')
                    if fnmatch.fnmatch(f, '*.csv')]

        # initialize input and output list
        feature_list_main = list()
        label_list_main = list()
        # train for surgical task
        for file in csv_list:
            # read the file into a data-frame
            df = pd.read_csv(source_path + input_folder +
                             surgery_name_list[surgery_selected] +
                             '/' + str(action_selected) +
                             '/' + file)

            # create motion windows and separate data into input and output
            # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)w
            feature_list, label_list = create_motion_windows(sliding_window, df)
            # create list of windows
            feature_list_main.extend(feature_list)
            label_list_main.extend(label_list)
        # reshape to train
        feature_list_main = np.reshape(feature_list_main, (len(feature_list_main), sliding_window, n_features))
        label_list_main = np.array(label_list_main)
        # get total number of batches
        total_motion_n_windows = len(feature_list_main)
        print("Total no. of motion windows for: " + surgery_name_list[surgery_selected][1:] +
              '- ' + str(action_selected) + ': ' + str(total_motion_n_windows))

        n_batches = int(total_motion_n_windows / batch_size)
        print("Total no. of batches for: " + surgery_name_list[surgery_selected][1:] +
              '- ' + str(action_selected) + ': ' + str(n_batches))

        # split data for training and testing
        x_train, x_test, y_train, y_test = train_test_split(feature_list_main, label_list_main, test_size=0.15, random_state=33)
        # train
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)
        # display summary of training
        model.summary()

        # plot losses
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Model loss - ' + surgery_name_list[surgery_selected] + ' - ' + str(action_selected))
        plt.ylabel('loss value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected][1:] + '_' + str(action_selected) + '_' + 'loss' + '.png',
                    dpi=300, bbox_inches='tight')

        # plot accuracies
        plt.plot(history.history['accuracy'], label='acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Model accuracy' + surgery_name_list[surgery_selected] + ' - ' + str(action_selected))
        plt.ylabel('accuracy value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected][1:] + '_' + str(action_selected) + '_' + 'acc' + '.png',
                    dpi=300, bbox_inches='tight')

        # test the model
        model.evaluate(x_test, y_test, batch_size=n_batches, verbose=2)

        # Create folder to save trained model
        os.mkdir(source_path + save_to_folder + save_model + '/' +
                 surgery_name_list[surgery_selected] + '_' + str(action_selected))
        # save the trained model
        model.save(source_path + save_to_folder + save_model +
                   '/' + surgery_name_list[surgery_selected] +
                   '_' + str(action_selected) + '/', save_format='tf')

