import pandas as pd
import numpy as np
from keras.models import model_from_json

import lstm
import data_processor

code = 7203
start = "2014/01/01"
end = "2016/12/31"
data_path = "data/"
result_path = "result/"

DATA_PERIOD = 20
NUM_PARAMS = 5
BATCH_SIZE = 32
NUM_EPOCH = 100


data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
data = data_processor.divide(data_processor.rate(data['CLOSE'].tolist()), DATA_PERIOD + 1)

model = lstm.create_lstm_model()
input_data = np.array(list(map(lambda n:n[:DATA_PERIOD], data))).reshape(len(data), DATA_PERIOD, 1)
output_data = np.array(list(map(lambda n:n[DATA_PERIOD], data))).reshape(len(data), 1)

model.fit(input_data, output_data, nb_epoch=NUM_EPOCH, batch_size=BATCH_SIZE)
lstm.save_model(model, code)
