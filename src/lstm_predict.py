import pandas as pd
import numpy as np
from keras.models import model_from_json

import lstm

code = 7203
start = "2014/01/01"
end = "2016/12/31"
data_path = "data/"
result_path = "result/"
DATA_PERIOD = 20

data = lstm.load_data(data_path + str(code) + ".csv", start, end)
data = lstm.divide(lstm.rate(data['CLOSE'].tolist()), DATA_PERIOD + 1)

test_data = np.array(list(map(lambda n:n[:DATA_PERIOD], data))[1200]).reshape(1, DATA_PERIOD, 1)
print(test_data)
model = lstm.load_model(result_path, code)
predicted = model.predict(test_data, verbose=1)
print(predicted)
