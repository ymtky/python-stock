import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from keras.models import model_from_json

import cnn
import data_processor
import cnn_data_generater

data_path = "data/"
model_path = "result/cnn/"

DATA_PERIOD = 20
DATA_DURATION = 5
NUM_PARAMS  = 5

def predict(code, start, end):
  test_data = cnn_data_generater.generate_predict_data(code, start, end)
  model = cnn.load_model(model_path, code)
  predicted = model.predict(test_data, verbose=1)
  predicted =  list(map(lambda n:n[0], predicted))

  data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
  data = data_processor.divide(data, DATA_PERIOD, 0)
  base_price = list(map(lambda n:n[DATA_PERIOD - 1], list(map(lambda n:n['CLOSE'].tolist(), data))))
  date = list(map(lambda n:(datetime.strptime(n[DATA_PERIOD - 1], "%Y-%m-%d") + timedelta(days = DATA_DURATION)).strftime("%Y-%m-%d"), list(map(lambda n:n['DATE'].tolist(), data))))

  df = pd.DataFrame({
    'date': date,
    'predict': predicted,
    'base_price': base_price
  })
  df.to_csv("result.csv", index=False)
  fig = (df.plot()).get_figure()
  fig.savefig('../figure.png', dpi=600)

def test(code, start, end):
  test_data = cnn_data_generater.generate_input_data(code, start, end)
  model = cnn.load_model(model_path, code)
  predicted = model.predict(test_data, verbose=1)
  predicted =  list(map(lambda n:n[0], predicted))

  data = data_processor.load_data(data_path + str(code) + ".csv", start, end)
  data = data_processor.divide(data, DATA_PERIOD + DATA_DURATION, 1)
  base_price = list(map(lambda n:n[DATA_PERIOD - 1], list(map(lambda n:n['CLOSE'].tolist(), data))))
  acual_price = list(map(lambda n:n[DATA_PERIOD + DATA_DURATION - 1], list(map(lambda n:n['CLOSE'].tolist(), data))))
  date = list(map(lambda n:n[DATA_PERIOD + DATA_DURATION - 1], list(map(lambda n:n['DATE'].tolist(), data))))

  df = pd.DataFrame({
    'date': date,
    'predict': predicted,
    'base_price': base_price,
    'acual_price': acual_price
  })
  df.to_csv("result.csv", index=False)

if __name__ == "__main__":
  code = 7203
  start = "2017-01-01"
  end = "2017-06-13"
  predict(code, start, end)
