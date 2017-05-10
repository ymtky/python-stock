import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.models import model_from_json

DATA_PERIOD = 20
NUM_PARAMS = 5
NUM_UNIT = 20
BATCH_SIZE = 32
NUM_EPOCH = 100

code = 7203
data_path = "data/"
result_path = "result/"

def load_data(file_path, start, end):
  df = pd.read_csv(file_path)
  df = df[df['DATE'] < end]
  df = df[df['DATE'] >= start]
  return df

def rate(data):
  result = []
  for i in range(len(data) - 1):
    result.append(round(data[i] / data[i + 1] - 1, 2))
  return result

def divide(data, period):
  result = []
  for i in range(len(data) - period):
    result.append(data[i: i + period])
  return result

def create_lstm_model():
  model = Sequential()
  model.add(LSTM(output_dim = NUM_UNIT, input_shape = (DATA_PERIOD, 1), return_sequences=False))
  model.add(Dense(1))
  model.add(Activation('linear'))
  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  return model

def save_model(model):
  model_json_str = model.to_json()
  open(result_path + 'model.json', 'w').write(model_json_str)
  model.save_weights(result_path + str(code) + '.h5');

def load_model(file_path, code):
  path = 'lstm/' + str(code)
  model = model_from_json(open(result_path + 'model.json').read())
  model.load_weights(result_path + str(code) + '.h5')
  return model
