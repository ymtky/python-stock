import numpy as np

import cnn
import data_processor
import cnn_data_generater

data_path = "data/"

def train(code, start, end):
  input_data = cnn_data_generater.generate_input_data(code, start, end)
  output_data = cnn_data_generater.generate_output_data(code, start, end)

  model = cnn.create_model()
  cnn.fit(model, input_data, output_data)
  cnn.save_model(model, code)

if __name__ == "__main__":
  code = 7203
  start = "2014-01-01"
  end = "2017-06-10"
  train(code, start, end)
