## --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import pandas as pd
import numpy as np
import random

# import tensorflow as tf
# from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPool1D, Lambda
# from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# File path to the database files
# source_path = os.getcwd() + '/../../Nihar/ML-data/SurgicalData'
source_path = os.getcwd() + "/../../../../Nihar/ML-data/SurgicalData"
surgery_selected = 0
#action_selected = 2

surgery_name_list = ['/Pericardiocentesis',
                     '/Thoracentesis']

data_folder = '5 Actions_10032020/PreparedData/'

input_folder = '/AnnotatedForSiamese'
save_model = '/06222021'
save_to_folder = '/Results/Siamese/AutoAnnotated'

## 1 ---  Define hyper parameters  ---  #
skill_levels = 3

no_sensors = 1
features_per_sensor = 13
n_features = no_sensors * features_per_sensor
n_classes = 3  # number of outputs for classification
epochs = 30

sliding_window = 150
window_step_size = 25
batch_size = 16


##  ---   Make motion windows   --- #

# return an input and output data frame with motion windows
def create_motion_windows(window_span, df_to_change, step_size):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - window_span)
    time_index = 0
    while time_index + window_span < len(df_to_change):
        a = df_to_change.iloc[time_index:time_index + window_span, :-1].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[time_index + window_span, 13:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
        time_index += step_size
    # for step in steps:  # range(len(df_to_change) - no_windows + 1):
    #     # print('Value of i is: {}'.format(i))
    #     a = df_to_change.iloc[step:step+no_windows, :-3].reset_index(drop=True).to_numpy()
    #     # a.reset_index(drop=True)
    #     b = df_to_change.iloc[step+no_windows, 13:].reset_index(drop=True).to_numpy()
    #     local_feature_df.append(a)
    #     local_label_df.append(b)
    return local_feature_df, local_label_df


# creating pairs from input data
# -----------------   LSTM ----------------------------------------- #
learning_rate = 0.001
n_units = 150 # number of lstm cells
## --- tf.Keras implementation of LSTM layers --- #
left_input_performance = Input((sliding_window, n_features))
right_input_performance = Input((sliding_window, n_features))


model = Sequential()
model.add(Conv1D(filters=38, kernel_size=2, activation='relu', input_shape=(sliding_window,n_features)))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=76, kernel_size=2, activation='relu'))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=156, kernel_size=2, activation='relu'))
model.add(MaxPool1D(pool_size=2, strides=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(468, activation='sigmoid'))

encoded_left = model(left_input_performance)
encoded_right = model(right_input_performance)
# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))

# Add the distance function to the network
L1_distance = L1_layer([encoded_left, encoded_right])

prediction = Dense(1, activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input_performance, right_input_performance], outputs=prediction)

optimizer = Adam(0.001, decay=2.5e-4)

siamese_net.summary()
siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# opt = tf.keras.optimizers.Adam(lr=learning_rate, momentum=0.9, decay=1e-2/epochs)

plot_model(siamese_net, show_shapes=True, show_layer_names=True)

## --- Training process  --- ##
print(' ------------- Training   ------------')
total_runs = 5
# Create folder to save all the plots
os.mkdir(source_path + save_to_folder + save_model + '/' + 'Graphs')

# train for all surgical procedures
# for surgery_selected in surgery_name_list[1]`:
for surgery_selected in range(0, len(surgery_name_list)):
    # train for each surgical task with procedure
    surgical_tasks = os.listdir(source_path + input_folder + surgery_name_list[surgery_selected] + '/')
    # for action_selected in surgical_tasks[0]:
    for action_selected in surgical_tasks:
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
            feature_list, label_list = create_motion_windows(sliding_window, df, window_step_size)
            # create list of windows
            feature_list_main.extend(feature_list)
            label_list_main.extend(label_list)

        # create input and output list
        sorting_list = list(zip(feature_list_main, label_list_main))

        # sort and count the windows based on labels
        count_experts = 0
        count_amateurs = 0
        expert_list = []
        amateur_list = []
        for _ in range(0, len(sorting_list)):
            if sorting_list[_][1] == 1:
                expert_list.append(list(sorting_list[_]))
                count_experts += 1
            else:
                amateur_list.append(list(sorting_list[_]))
                count_amateurs += 1

        # remove the excess windows to keep same number of input data per label
        if count_experts > count_amateurs:
            excess_samples = count_experts - count_amateurs
            for i in range(excess_samples):
                expert_list.pop(random.randint(0, len(expert_list) - 1))
        else:
            excess_samples = count_amateurs - count_experts
            for i in range(excess_samples):
                amateur_list.pop(random.randint(0, len(amateur_list) - 1))

        # drop all the previous labels
        for i in range(len(expert_list)):
            expert_list[i] = np.asarray(expert_list[i][0])
            amateur_list[i] = np.asarray(amateur_list[i][0])

        # shuffle samples
        random.seed(7)
        random.shuffle(expert_list)
        random.shuffle(amateur_list)

        # creating pairs to train model
        left_input = []
        right_input = []
        targets = []
        # number of pairs per image
        pairs = 2
        # get length of input data
        len_individual_input = len(expert_list)
        # create pairs with labels
        for i in range(0, len(expert_list)):
            for _ in range(pairs):
                compare_to = i
                # making sure not to pair with itself
                while compare_to == i:
                    compare_to = random.randint(0, len(expert_list)-1)
                # create positive pair
                left_input.append(expert_list[i])
                right_input.append(expert_list[compare_to])
                targets.append(1.)
                # create negative pair
                left_input.append(expert_list[i])
                right_input.append(amateur_list[compare_to])
                targets.append(0.)

        # combine to shuffle
        data_pairs = list(zip(left_input, right_input, targets))
        random.shuffle(data_pairs)

        # split data for testing
        test_list = []
        for i in range(0, int(len(data_pairs)/10)):
            idx = random.randint(0,len(data_pairs)-1)
            test_list.append(data_pairs[idx])
            data_pairs.pop(idx)

        # split input pairs and labels
        left_input, right_input, targets = list(zip(*data_pairs))
        test_left, test_right, test_targets = list(zip(*test_list))

        # reshape to input for model
        left_input = np.reshape(left_input, (len(left_input), sliding_window, n_features))
        right_input = np.reshape(right_input, (len(right_input), sliding_window, n_features))
        targets = np.array(targets)
        test_left = np.reshape(test_left, (len(test_left), sliding_window, n_features))
        test_right = np.reshape(test_right, (len(test_right), sliding_window, n_features))
        test_targets = np.array(test_targets)

        siamese_net.summary()
        history = siamese_net.fit([left_input, right_input], targets,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=([test_left, test_right], test_targets))

        # Create folder to save trained model
        os.mkdir(source_path + save_to_folder + save_model + '/' +
                 surgery_name_list[surgery_selected] + '_' + str(action_selected))
        # save the trained model
        siamese_net.save(source_path + save_to_folder + save_model +
                         '/' + surgery_name_list[surgery_selected] +
                         '_' + str(action_selected) + '/', save_format='tf')

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Model loss - ' + surgery_name_list[surgery_selected] + ' - ' + str(action_selected))
        plt.ylabel('loss value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected] + '_' + str(action_selected) + '_' + 'loss' + '.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        plt.plot(history.history['accuracy'], label='acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Model accuracy - ' + surgery_name_list[surgery_selected] + ' - ' + str(action_selected))
        plt.ylabel('accuracy value')
        plt.xlabel('epoch')
        plt.legend(loc="upper left")
        plt.savefig(source_path + save_to_folder + save_model + '/' + 'Graphs' + '/' +
                    surgery_name_list[surgery_selected] + '_' + str(action_selected) + '_' + 'acc' + '.png',
                    dpi=300, bbox_inches='tight')
        plt.show()



