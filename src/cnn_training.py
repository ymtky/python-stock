import numpy as np

import cnn
import data_processor

code = 7203
start = "2014/01/01"
end = "2016/12/31"
data_path = "data/"

DATA_PERIOD = 20

data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
data = data_processor.divide(data['CLOSE'].tolist(), DATA_PERIOD + 1, DATA_PERIOD + 1)

model = cnn.create_model()
input_data = np.array(list(map(lambda n:data_processor.normarize(n[:DATA_PERIOD]), data))).reshape(len(data), DATA_PERIOD, 1)
output_data = np.array(list(map(lambda n:n[DATA_PERIOD] / min(n[:DATA_PERIOD]), data))).reshape(len(data), 1)

print(output_data)

cnn.fit(model, input_data, output_data)
cnn.save_model(model, code)
