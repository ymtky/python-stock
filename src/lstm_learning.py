import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

DATA_PERIOD = 20
NUM_PARAMS = 5
NUM_UNIT = 16
BATCH_SIZE = 32
NUM_EPOCH = 50

code = 7203
start = "2014/01/01"
end = "2016/12/31"
path = "data/"

def load(file_path, start, end):
  df = pd.read_csv(file_path)
  df = df[df['DATE'] < end]
  df = df[df['DATE'] >= start]
  return df

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
  model.compile(loss='mape', optimizer='adam', metrics=['accuracy'])
  return model

data = load(path + str(code) + ".csv", start, end)
data = divide(data['CLOSE'].tolist(), DATA_PERIOD + 1)

model = create_lstm_model()
input_data = np.array(list(map(lambda n:n[:DATA_PERIOD], data))).reshape(len(data), DATA_PERIOD, 1)
output_data = np.array(list(map(lambda n:n[DATA_PERIOD], data))).reshape(len(data), 1)

model.fit(input_data, output_data, nb_epoch=NUM_EPOCH, batch_size=BATCH_SIZE)
