# --- Libraries   ---  #
# File imports and aggregates data from multiple databases
import os
import fnmatch
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import csv

# File path to the database files
source_path = os.getcwd() + '/../..' + '/Data/'
data_folder = '4sensors_6actions_dataset/Prepared Data/'
#data_folder_test = 'Six_Actions_Updated_Test/'


## ---   Fetch and split data into train and test   --- #
no_of_actions = 6
csv_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]

test_list = []
test_subjects = []
train_list = []
# randomly select a subject for testing network
for i in range(0, 2):
    x = random.randint(1, int(len(csv_list)/no_of_actions) - 1)
    if x < 10:
        x = '0' + str(x)
        test_subjects.append(x)
    else:
        test_subjects.append(str(x))
# separate data into test and train
for i in test_subjects:
    for f in csv_list:
        if fnmatch.fnmatch(f, '*' + i + '.csv'):
            test_list.append(f)
        else:
            train_list.append(f)

##  ---  Define hyper parameters  ---  #
no_sensors = 2
features_per_sensor = 6
n_features = no_sensors * features_per_sensor
n_classes = 6 #number of outputs for classification
epochs = 10
learning_rate = 0.1

n_units = 100 # number of lstm cells

##  ---   Make motion windows   --- #
sliding_window_1 = 60
sliding_window_2 = [60, 80, 100, 120]
batch_size = 25


# return an input and output data frame with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for step in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[step:step+no_windows, :-6].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[step+no_windows, 12:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df


##  ---  Create placeholders for features and labels  --- #
# input_data = tf.placeholder(tf.float32, [None, None, n_features])
input_data = tf.placeholder(tf.float32, [None, sliding_window_1, n_features], name='input_ph')
output_data = tf.placeholder(tf.float32, [None, n_classes], name='output_ph')

# weight = tf.Variable(tf.truncated_normal([n_units, int(output_data.get_shape()[1])]))
weight = tf.Variable(tf.zeros([n_units, int(output_data.get_shape()[1])]), name='w0')
bias = tf.Variable(tf.zeros([output_data.get_shape()[1]]), name='b0')

## ---   Create LSTM cells and Prediction   --- #

cell = tf.nn.rnn_cell.LSTMCell(n_units, state_is_tuple=True, name='cell')  # 6n return output and cell state
# unroll network and store output and state
# val_output dim [batch_size, window_size, features]
val_output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
# get the last value from the output
val = tf.transpose(val_output, [1, 0, 2])
# last = tf.gather(val, val.get_shape()[0] - 1)
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# ---- Calculate Prediction  --- #
# prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
prediction = tf.matmul(last, weight) + bias

# calculate the error in prediction
# cross_entropy = -tf.reduce_sum(output_data * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_data))

# select an optimizer and pass the error value to be minimized
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
minimize = optimizer.minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(output_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#mistakes = tf.not_equal(tf.argmax(output_data,1), tf.argmax(prediction,1))
#error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
saver = tf.train.Saver()

# execute the model in a session
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

## ----  Creating a dataframe to understand errors / batch(file)  _---- #
error_matrix = np.zeros((epochs, len(train_list)))
error_dataframe = pd.DataFrame(error_matrix, columns=train_list)
#error_dataframe.at[1, train_list[0]] = 10

epoch_acc = []
## --- Training process  ---  #

print(' ------------- Training   ------------')
for i in range(epochs):
    j = 0
    avg_acc = 0
    random.shuffle(train_list)
    feature_list_main = list()
    label_list_main = list()
    for file in train_list:
        # read the file into a data-frame
        df = pd.read_csv(source_path + data_folder + file)
        # drop all Time columns
        df = df.drop(df.columns[[0, 4, 8, 12, 16]], axis=1)

        # create motion windows and separate data into input and output
        # feature_list, label_list = create_motion_windows(random.choice(sliding_window_2), df)
        feature_list, label_list = create_motion_windows(sliding_window_1, df)
        feature_list_main.extend(feature_list)
        label_list_main.extend(label_list)

    # find number of batches per file
    total_motion_n_windows = len(feature_list_main)
    n_batches = int(total_motion_n_windows / batch_size)

    # -- shuffle all motion window inputs and outputs -- #
    shuffle_list = list(zip(feature_list_main, label_list_main))
    random.shuffle(shuffle_list)
    feature_list_main, label_list_main = zip(*shuffle_list)
    for batch_n in range(n_batches):
        # # -- shuffle all motion window inputs and outputs -- #
        # shuffle_list = list(zip(feature_list_main, label_list_main))
        # random.shuffle(shuffle_list)
        # feature_list_main, label_list_main = zip(*shuffle_list)

        # create batches of features and labels
        curr_index = batch_n * batch_size
        feature_list = feature_list_main[curr_index:curr_index + batch_size - 1]
        label_list = label_list_main[curr_index:curr_index + batch_size - 1]
        sess.run(minimize, {input_data: feature_list, output_data: label_list})
        # err, cr_ent = sess.run([error,cross_entropy], {input_data: feature_list, output_data: label_list})
        # print('Epoch No: {} Batch No: {}        Error: {}  and  Cross_entropy: {}'.format(i+1, j+1, err, cr_ent))
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={input_data: feature_list, output_data: label_list})

        # print('Epoch No: {} Batch No: {}        Acc: {}  and  loss: {}'.format(i + 1, j + 1, acc, loss))
        # print('Correct_pred: '.format(correct_pred))
        avg_acc += acc
        # error_dataframe.at[i, file] = acc
        # print('Error: {}  and  Cross_entropy: {}'.format(err, cr_ent))
        j += 1
    epoch_acc.append(avg_acc/n_batches)
    print('Epoch No: {} Accuracy: {}'.format(i+1,epoch_acc[i]))
    saver.save(sess, source_path + 'Saved/TF2_08122020/lstm_peek_12F6L_08122020')

##print('------  Validation  -------------')
# sliding_window_test = 120
#tf.saved_model.load(source_path + 'Saved/my-test-model')
random.shuffle(test_list)
for file in test_list:
    df = pd.read_csv(source_path + data_folder + file)
    # drop all Time columns
    df = df.drop(df.columns[[0, 4, 8, 12, 16]], axis=1)
    # create motion windows and separate data into input and output
    feature_df, label_df = create_motion_windows(sliding_window_1, df)

    #err, cr_ent = sess.run([error, cross_entropy], {input_data: feature_df, output_data: label_df})
    #print('Error: {}  and  Cross_entropy: {}'.format(err, cr_ent))

    acc, loss = sess.run([accuracy, cross_entropy], feed_dict={input_data: feature_df, output_data: label_df})
    print('File: {}  Acc: {} '.format(file, acc))