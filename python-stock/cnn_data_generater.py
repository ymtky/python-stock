import numpy as np
import data_processor

DATA_PERIOD = 20
DATA_DURATION = 5

data_path = "data/"

def generate_input_data(code, start, end):
  data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
  data = data_processor.divide(data, DATA_PERIOD + DATA_DURATION, 1)
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

def generate_output_data(code, start, end):
  data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
  data = data_processor.divide(data, DATA_PERIOD + DATA_DURATION, 1)
  output_data = np.array(list(map(lambda n:n[DATA_PERIOD + DATA_DURATION - 1] / n[DATA_PERIOD - 1], list(map(lambda n:n['CLOSE'].tolist(), data)) ))).reshape(len(data), 1)
  return output_data

def generate_predict_data(code, start, end):
 data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
 data = data_processor.divide(data, DATA_PERIOD, 0)
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
