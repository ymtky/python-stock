import pandas as pd
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, Flatten, Convolution1D
from keras.layers.pooling import MaxPooling1D

DATA_PERIOD = 20
NUM_PARAMS = 5
NUM_UNIT = 100
BATCH_SIZE = 32
NUM_EPOCH = 100

result_path = "result/cnn/"

def create_model():
  model = Sequential()
  model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', input_shape = (DATA_PERIOD, 1)))
  model.add(Activation('relu'))
  model.add(MaxPooling1D(pool_length=2))
  model.add(Convolution1D(nb_filter=64, filter_length=3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling1D(pool_length=2))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(250, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(1, activation='linear'))
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  return model

def save_model(model, code):
  model_json_str = model.to_json()
  open(result_path + 'model.json', 'w').write(model_json_str)
  model.save_weights(result_path + str(code) + '.h5');

def fit(model, input_data, output_data):
  model.fit(input_data, output_data, nb_epoch=NUM_EPOCH, batch_size=BATCH_SIZE)

def load_model(file_path, code):
  model = model_from_json(open(result_path + 'model.json').read())
  model.load_weights(result_path + str(code) + '.h5')
  return model
