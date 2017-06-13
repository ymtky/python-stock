import numpy as np

import cnn
import data_processor
import cnn_data_processor

code = 7203
start = "2014-01-01"
end = "2017-06-10"
data_path = "data/"

DATA_PERIOD = 20
DATA_DURATION = 5

def get_input_data(data):
  value_data = list(map(lambda n:n.loc[:,['CLOSE', 'OPEN', 'HIGH', 'LOW']].values, data))
  value_data = list(map(lambda n:n[:DATA_PERIOD], value_data))
  value_data = data_processor.normarize2D(np.array(value_data))

  volume_data = list(map(lambda n:n['VOLUME'].tolist(), data))
  volume_data = list(map(lambda n:data_processor.normarize(n), volume_data))

  for i in range(len(value_data)):
    for j in range(len(value_data[i])):
      value_data[i][j].append(volume_data[i][j])

  value_data = np.array(value_data).reshape(len(value_data), DATA_PERIOD, 5)
  return value_data

def get_output_data(data):
  output_data = np.array(list(map(lambda n:n[DATA_PERIOD + DATA_DURATION - 1] / n[DATA_PERIOD - 1], list(map(lambda n:n['CLOSE'].tolist(), data)) ))).reshape(len(data), 1)
  return output_data

data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
data = data_processor.divide(data, DATA_PERIOD + DATA_DURATION, 1)
input_data = cnn_data_processor.generate_input_data(data)
output_data = cnn_data_processor.generate_output_data(data)

model = cnn.create_model()
cnn.fit(model, input_data, output_data)
cnn.save_model(model, code)
