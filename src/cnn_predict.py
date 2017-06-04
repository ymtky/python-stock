import pandas as pd
import numpy as np
from keras.models import model_from_json

import cnn
import data_processor

code = 7203
start = "2014/01/01"
end = "2016/12/31"
data_path = "data/"
model_path = "result/cnn/"

DATA_PERIOD = 20

data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
data = data_processor.divide(data['CLOSE'].tolist(), DATA_PERIOD + 1, 1)

test_data = np.array(list(map(lambda n:data_processor.normarize(n[:DATA_PERIOD]), data))).reshape(len(data), DATA_PERIOD, 1)

print(test_data)

model = cnn.load_model(model_path, code)
predicted = model.predict(test_data, verbose=1)

print(predicted)
