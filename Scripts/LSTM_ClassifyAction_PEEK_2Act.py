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

# File path to the database files
source_path = os.getcwd() + '/Data/'
data_folder = 'Combined_CSV_1/'

## ---   Fetch and split data into train and test   --- #
csv_list_1 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*S.csv')]
csv_list_2 = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*C.csv')]
no_of_actions = 2
csv_list = [f for f in os.listdir(source_path + data_folder) if fnmatch.fnmatch(f, '*.csv')]
#print(range(0,int(len(csv_list)/2) - 1))
test_subjects = []
performances_to_test = 1

for i in range(0,2):
    x = random.randint(00, int(len(csv_list)/no_of_actions) - 1)
    test_subjects.append(x)

test_list = []
train_list = []
# Create training and testing lists of files
# All actions included in the performance are held for testing
for i in range(0,len(test_subjects)):
    if test_subjects[i] - 9 <= 0:
        for f in os.listdir(source_path + data_folder):
            if fnmatch.fnmatch(f, '*0{}*.csv'.format(test_subjects[i])):
                test_list.append(f)
    else:
        for f in os.listdir(source_path + data_folder):
            if fnmatch.fnmatch(f, '*{}*.csv'.format(test_subjects[i])):
                test_list.append(f)

train_list = [x for x in csv_list if x is not test_list]

##   ----   Fetch the stats file   ----   #
stats_file = open(os.getcwd()+'/Results/Stats.csv','w')


##  ---   Normalize the data  --- #

# min and max values for each feature
pos_x = (20,-20)
pos_y = (22,-10)
pos_z = (50,-50)
ang_x = (2,-2)
ang_z = (4,2)
ang_y = (2,-2)
vel_x = (25,-10)
vel_y = (15,-10)
vel_z = (25,-10)
vel_ang = (0.4,-0.4)

#  input msx and min to normalize data
def normalize_column_custom(max, min, input_df, feature_list):
    df_copy = input_df.copy()
    for feature in feature_list:
        df_copy[feature] = df_copy[feature].map(lambda a: scale_input(a, max, min))
        # df_copy[feature] = (input_df[feature] - min) / (max - min)
    return df_copy


def scale_input(x, max, min):
    if x >= max:
        return 1
    elif x <= min:
        return 0
    else:
        return (x - min)/(max - min)


# use max and min values in sequence to normalize data
def normalize_column(input_df, feature_name):
    df_copy = input_df.copy()
    for feature in feature_name:
        max_value = df_copy[feature].max()
        min_value = df_copy[feature].min()
        if max_value == min_value:
            print("Error: Cannot normalize when max and min values are equal")
            return df_copy
        df_copy[feature] = (df[feature] - min_value) / (max_value - min_value)
    return df_copy



##  ---  data info   ---   #
n_features = 12
#batch_size = 1
n_classes = 2 #number of outputs for classification


##  ---  Define hyper parameters  ---  #
epochs = 10
learning_rate = 0.1
n_units = 100 # number of lstm cells
sliding_window = 60
batch_size = 20


##  ---   Make motion windows   --- #
# return an input and output dataframe with motion windows
def create_motion_windows(no_windows, df_to_change):
    local_feature_df = []
    local_label_df = []
    steps = range(len(df_to_change) - no_windows)
    for i in steps:#range(len(df_to_change) - no_windows + 1):
        # print('Value of i is: {}'.format(i))
        a = df_to_change.iloc[i:i+no_windows, :-3].reset_index(drop=True).to_numpy()
        # a.reset_index(drop=True)
        b = df_to_change.iloc[i+no_windows, 13:].reset_index(drop=True).to_numpy()
        local_feature_df.append(a)
        local_label_df.append(b)
    return local_feature_df, local_label_df


##  ---  Create placeholders for features and labels  --- #
input_data = tf.placeholder(tf.float32, [None, sliding_window, n_features])
output_data = tf.placeholder(tf.float32, [None, 2])


# weight = tf.Variable(tf.truncated_normal([n_units, int(output_data.get_shape()[1])]))
weight = tf.Variable(tf.zeros([n_units, int(output_data.get_shape()[1])]))
bias = tf.Variable(tf.zeros([output_data.get_shape()[1]]))

## ---   Create LSTM cells and Prediction   --- #

cell = tf.nn.rnn_cell.LSTMCell(n_units, state_is_tuple=True) # 6n return output and cell state
# unroll network and store output and state
# val_output dim [batch_size, window_size, features]
val_output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
# get the last value from the output
val = tf.transpose(val_output, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

#save the training parameters
saver = tf.train.Saver()
# ---- Calculate Prediction  --- #
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

# execute the model in a session
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

## ----  Creating a dataframe to understand errors / batch(file)  _---- #
error_matrix = np.zeros((epochs, len(train_list)))
error_dataframe = pd.DataFrame(error_matrix, columns=train_list)
#error_dataframe.at[1, train_list[0]] = 10



# run training for no_epochs
# random.shuffle(train_list)
feature_list_main = list()
label_list_main = list()
for file in train_list:
    # read the file into a data-frame
    df = pd.read_csv(source_path + data_folder + file,
                     usecols=['X', 'Y', 'Z', 'A', 'B', 'G', 'Vx', 'Vy', 'Vz',
                              'Va', 'Vb', 'Vg', 'Time', 'Square', 'Circle'])
    # normalize the data
    df = normalize_column_custom(pos_x[0], pos_x[1], df, ['X'])
    df = normalize_column_custom(pos_y[0], pos_y[1], df, ['Y'])
    df = normalize_column_custom(pos_z[0], pos_z[1], df, ['Z'])
    df = normalize_column_custom(ang_x[0], ang_x[1], df, ['A', 'B'])
    df = normalize_column_custom(ang_z[0], ang_z[1], df, 'G')
    df = normalize_column_custom(vel_x[0], vel_x[1], df, ['Vx'])
    df = normalize_column_custom(vel_y[0], vel_y[1], df, ['Vy'])
    df = normalize_column_custom(vel_z[0], vel_z[1], df, ['Vz'])
    df = normalize_column_custom(vel_ang[0], vel_ang[1], df, ['Va', 'Vb', 'Vg'])

    # create motion windows and separate data into input and output
    feature_list, label_list = create_motion_windows(sliding_window, df)
    feature_list_main.extend(feature_list)
    label_list_main.extend(label_list)

total_motion_n_windows = len(feature_list_main)
n_batches = int(total_motion_n_windows / batch_size)



##   ----   Training    ----    #

print(' ------------- Training   ------------')
for i in range(epochs):
    j = 0
    acc_avg = 0
    shuffle_list = list(zip(feature_list_main, label_list_main))
    random.shuffle(shuffle_list)
    feature_list_main, label_list_main = zip(*shuffle_list)
    for batch_n in range(n_batches):
        # batch_size = len(feature_list_main)
        # inp, out = feature_df[0:batch_size], label_df[0:batch_size]
        curr_index = batch_n * batch_size
        feature_list = feature_list_main[curr_index:curr_index + batch_size - 1]
        label_list = label_list_main[curr_index:curr_index + batch_size - 1]
        sess.run(minimize, {input_data: feature_list, output_data: label_list})
        # err, cr_ent = sess.run([error,cross_entropy], {input_data: feature_list, output_data: label_list})
        # print('Epoch No: {} Batch No: {}        Error: {}  and  Cross_entropy: {}'.format(i+1, j+1, err, cr_ent))
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={input_data: feature_list, output_data: label_list})

        #print('Epoch No: {} Batch No: {}        Acc: {}  and  loss: {}'.format(i + 1, j + 1, acc, loss))
        #print('Correct_pred: '.format(correct_pred))
        # error_dataframe.at[i, file] = acc
        # print('Error: {}  and  Cross_entropy: {}'.format(err, cr_ent))
        acc_avg += acc
        j += 1

    acc_avg = acc_avg/j
    print('Epoch No: {} Acc: {} '.format(i + 1, acc_avg))
    save_path = saver.save(sess, os.getcwd()+'/Results/model.ckpt')

print('------  Validation  -------------')
test_acc = 0
for file in test_list:
    df = pd.read_csv(source_path + data_folder + file, index_col=0)
    # normalize the data
    df = normalize_column_custom(pos_x[0], pos_x[1], df, ['X'])
    df = normalize_column_custom(pos_y[0], pos_y[1], df, ['Y'])
    df = normalize_column_custom(pos_z[0], pos_z[1], df, ['Z'])
    df = normalize_column_custom(ang_x[0], ang_x[1], df, ['A', 'B'])
    df = normalize_column_custom(ang_z[0], ang_z[1], df, 'G')
    df = normalize_column_custom(vel_x[0], vel_x[1], df, ['Vx'])
    df = normalize_column_custom(vel_y[0], vel_y[1], df, ['Vy'])
    df = normalize_column_custom(vel_z[0], vel_z[1], df, ['Vz'])
    df = normalize_column_custom(vel_ang[0], vel_ang[1], df, ['Va', 'Vb', 'Vg'])

    # create motion windows and separate data into input and output
    feature_df, label_df = create_motion_windows(sliding_window, df)

    #err, cr_ent = sess.run([error, cross_entropy], {input_data: feature_df, output_data: label_df})
    #print('Error: {}  and  Cross_entropy: {}'.format(err, cr_ent))

    acc, loss = sess.run([accuracy, cross_entropy], feed_dict={input_data: feature_df, output_data: label_df})
    test_acc += acc
    print('File:{}   Acc: {}'.format(file, acc))
print('Average accuracy: {}'.format(test_acc/len(test_list)))