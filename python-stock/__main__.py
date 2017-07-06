import sys
from datetime import datetime, timedelta

import get_data
import cnn_training
import cnn_predict

if __name__ == '__main__':
  code = str(sys.argv[1])
  date = datetime.now()
  get_data.get_daily_data(code, datetime(2014, 1, 1), datetime.now())
  cnn_training.train(code, "2014-01-01",date.strftime("%Y-%m-%d"))
  cnn_predict.predict(code, (date - timedelta(days = 100)).strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
