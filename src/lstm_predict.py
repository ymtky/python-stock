import pandas as pd
import numpy as np
from keras.models import model_from_json

code = 7203
start = "2014/01/01"
end = "2016/12/31"
data_path = "data/"
result_path = "result/"
DATA_PERIOD = 20

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

def load_weight(file_path, code):
  path = 'lstm/' + str(code)
  model = model_from_json(open(result_path + 'model.json').read())
  model.load_weights(result_path + str(code) + '.h5')
  return model

data = load(data_path + str(code) + ".csv", start, end)
data = divide(data['CLOSE'].tolist(), DATA_PERIOD + 1)

test_data = np.array(list(map(lambda n:n[:DATA_PERIOD], data))[0]).reshape(1, DATA_PERIOD, 1)
model = load_weight(result_path, code)
predicted = model.predict(test_data, verbose=1)
print(predicted)
