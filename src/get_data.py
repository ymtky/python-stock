import requests
from datetime import datetime, timedelta
import pandas as pd

url = "https://www.google.com/finance/getprices"
lsat_date = datetime.now() #データの取得開始日
interval = 86400  #データの間隔(秒)。1日 = 86400秒
market = "TYO"  #取引所のコード　TYO=東京証券取引所
period =  "3Y" #データを取得する期間

def get_daily_data(code, from_date: datetime, to_date: datetime):
  peropd_days = (to_date - from_date).days

  params = {
    'q': code,
    'i': interval,
    'x': market,
    'p': period,
    'ts': to_date.timestamp()
  }

  r = requests.get(url, params=params)

  lines = r.text.splitlines()
  columns = lines[4].split("=")[1].split(",")
  pridces = lines[8:]

  base_date = datetime.fromtimestamp(int(pridces[0].split(",")[0].lstrip('a')))
  result = []

  for price in pridces[0:]:
    cols = price.split(",")
    #dateがタイムスタンプの場合はdatetimeに
    if(not cols[0].isdigit()):
      base_date = datetime.fromtimestamp(int(cols[0].lstrip('a')))
      result.append([base_date.date(), cols[1], cols[2], cols[3], cols[4], cols[5]])
    else:
      result.append([(base_date + timedelta(days = int(cols[0]))).date(), cols[1], cols[2], cols[3], cols[4], cols[5]])

  df = pd.DataFrame(result, columns = columns)
  data_path = "data/"
  df.to_csv(data_path + str(code) + ".csv", index=False)

get_daily_data(7203, datetime(2014, 1, 1), datetime.now())
