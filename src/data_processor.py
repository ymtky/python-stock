import pandas as pd
import numpy as np

data_path = "data/"

def load_data(file_path, start, end):
  df = pd.read_csv(file_path)
  df = df[df['DATE'] < end]
  df = df[df['DATE'] >= start]
  return df

def rate(data):
  result = []
  for i in range(len(data) - 1):
    result.append(round(data[i] / data[i + 1] - 1, 2))
  return result

#分割する
def divide(data, period, space):
  result = []
  for i in range(len(data) - period):
    result.append(data[i: i + period])
  return result[::space]

#0~1に変換する
def normarize(data):
  result = []
  mx = max(data)
  mn = min(data)
  for i in data:
    result.append((i - mn) / (mx - mn))
  return result
