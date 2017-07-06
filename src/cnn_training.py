import numpy as np

import cnn
import data_processor
import cnn_data_processor

data_path = "data/"

DATA_PERIOD = 20
DATA_DURATION = 7

def train(code, start, end):
  data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
  data = data_processor.divide(data, DATA_PERIOD + DATA_DURATION, 1)
  input_data = cnn_data_processor.generate_input_data(data)
  output_data = cnn_data_processor.generate_output_data(data)

  model = cnn.create_model()
  cnn.fit(model, input_data, output_data)
  cnn.save_model(model, code)

if __name__ == "__main__":
    train(7203, "2014-01-01", "2017-06-10")
