import pandas as pd
import numpy as np
from keras.models import model_from_json

import cnn
import data_processor
import cnn_data_processor

code = 7203
start = "2017-01-01"
end = "2017-06-13"
data_path = "data/"
model_path = "result/cnn/"

DATA_PERIOD = 20
DATA_DURATION = 5
NUM_PARAMS  = 5

data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
data = data_processor.divide(data, DATA_PERIOD + DATA_DURATION, 1)
test_data = cnn_data_processor.generate_input_data(data)

model = cnn.load_model(model_path, code)
predicted = model.predict(test_data, verbose=1)
predicted =  list(map(lambda n:n[0], predicted))
base_price = list(map(lambda n:n[DATA_PERIOD - 1], list(map(lambda n:n['CLOSE'].tolist(), data))))
acual_price = list(map(lambda n:n[DATA_PERIOD + DATA_DURATION - 1], list(map(lambda n:n['CLOSE'].tolist(), data))))

df = pd.DataFrame({
  'predict': predicted,
  'base_price': base_price,
  'acual_price': acual_price
})
df.to_csv("result.csv", index=False)
