# Class that trains a CNN using kinematic data from the JIGSAWS dataset

import numpy as np
import pandas as pd
import tensorflow as tf


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import GlobalAvgPool1D, Flatten


class 1D_CNN:

def __init__(self):
	
	self.model = Sequential()
	
	self.learning_rate = 0.01
	self.epochs = 10
	self.batch_size = 50
	
	self.window_size = 0
	self.window_step_size = 1



def createMode(self,time_steps, n_features, n_classes, filters, kernel_size, pool_size, strides, dropout_value, learning_rate):
	
	self.model.add(Conv1D(filters=38, kernel_size=2, activation='relu', input_shape=(180, n_features)))
	self.model.add(MaxPool1D(pool_size=2, strides=2))
	self.model.add(Dropout(0.2))
	self.model.add(Conv1D(filters=76, kernel_size=2, activation='relu'))
	self.model.add(MaxPool1D(pool_size=2, strides=2))
	self.model.add(Dropout(0.2))
	self.model.add(Conv1D(filters=152, kernel_size=2, activation='relu'))
	self.model.add(MaxPool1D(pool_size=2, strides=2))
	self.model.add(Dropout(0.2))
	self.model.add(Flatten())
	self.model.add(Dropout(0.5))
	self.model.add(Dense(64, activation='relu'))
	self.model.add(Dropout(0.5))
	self.model.add(Dense(32, activation='relu'))
	self.model.add(Dense(n_classes, activation='softmax'))
	opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-3)

	self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
