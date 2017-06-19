import numpy as np

import cnn
import data_processor
import cnn_data_processor

code = 7203
start = "2014-01-01"
end = "2017-06-10"
data_path = "data/"

DATA_PERIOD = 20
DATA_DURATION = 7

data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
data = data_processor.divide(data, DATA_PERIOD + DATA_DURATION, 1)
input_data = cnn_data_processor.generate_input_data(data)
output_data = cnn_data_processor.generate_output_data(data)

model = cnn.create_model()
cnn.fit(model, input_data, output_data)
cnn.save_model(model, code)
