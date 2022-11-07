########## CONFIG ##########
### Imports
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import requests
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

### Training Data Specs
# stock symbols to train on
trainingStock = 'TSLA'#['TSLA', 'AAPL', 'GOOG', 'MSFT']
# NOTE: making this longer than 5 items will cause some to 
# automatically use CSV data if available, as the API has a limit 
# of 5 queries per minute at free tier

# column to use as training data
trainingColumns = ["high", "low", "open", "close", "volume"]

# time frame to get training data for
trainStart = datetime.datetime(2010, 1, 1)
trainEnd = datetime.datetime(2022, 10, 30)

########## END CONFIG ##########


# data retrieval class
class StockReader():

  def __init__(self, symbol=None, start=None, end=None):
    self.symbol = symbol

    # initialize start/end dates if not provided
    if end is None:
      end = datetime.datetime.today()
    if start is None:
      start = datetime.datetime.today() - relativedelta(years=10)

    # if no symbol provided, default to Tesla
    if symbol is None:
      symbol = 'TSLA'

    self.start = start
    self.end = end

    # convert dates to strings
    startString = self.start.strftime('%Y-%m-%d')
    endString = self.end.strftime('%Y-%m-%d')

    url = 'https://api.polygon.io/v2/aggs/ticker/{}/range/1/day/{}/{}?apiKey={}'
    key = '2Uyf_Dr3xhnccB16gtsfkbq15SELtLgz'
    self.url = url.format(self.symbol, startString, endString, key)

  def read(self):

    r = requests.get(self.url)
    try:
      df = pd.DataFrame(r.json()['results'])

      df.insert(0, 'symbol', self.symbol)

      df = df[['symbol', 't', 'h', 'l', 'o', 'c', 'v']]
      df = df.set_index('t')

      # rename columns to be more human friendly
      df.rename(columns={
        't': 'date',
        'h': 'high',
        'l': 'low',
        'o': 'open',
        'c': 'close',
        'v': 'volume'
      },
                inplace=True)

      # backup data to CSV
      df.to_csv(self.symbol + ".csv")

      return df
    except Exception as err:
      print(self.url, err)
      # if an error is encountered, use most recent CSV backup
      print(f"Could not retrieve data for {self.symbol}, using backup data")
      try:
        trainingData.append(pd.read_csv(self.symbol + ".csv"))
      except:
        print(f"No backup data found for {self.symbol}")
      return None



reader = StockReader(symbol=trainingStock, start=trainStart, end=trainEnd)
trainingData = reader.read()

# plot training data
#ax = trainingData.plot(x='date', y='high')
#plt.show(block=False)

# pause before continuing
#input("Displaying training data. Press enter to continue.")

#trainingData.drop(['date'], axis=1)

# generate column to train predictions for
trainingData["Target"] = trainingData.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["close"]

# initialize and set parameters for model
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

# for testing: train on every certain number of entries, then test on the next bit and see how accurate it is so far
trainLength = 50
testLength = 25

predictions = []

def backtest(data, model, predictors, start=10, step=5):
  # calculate rolling means
  weekly_mean = data.rolling(7).mean()["close"]
  quarterly_mean = data.rolling(90).mean()["close"]
  annual_mean = data.rolling(365).mean()["close"]
  weekly_trend = data.shift(1).rolling(7).sum()["Target"]

  # add to data the ratio between the daily close and the rolling means
  data["weekly_mean"] = weekly_mean / data["close"]
  data["quarterly_mean"] = quarterly_mean / data["close"]
  data["annual_mean"] = annual_mean / data["close"]

  # add to data ratios between rolling means
  data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]

  data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]

  # add weekly trend raw
  data["weekly_trend"] = weekly_trend

  # add ratios between other values
  data["open_close_ratio"] = data["open"] / data["close"]
  data["high_close_ratio"] = data["high"] / data["close"]
  data["low_close_ratio"] = data["low"] / data["close"]

  # update list of training columns
  fullTrainingColumns = trainingColumns + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "weekly_trend", "open_close_ratio", "high_close_ratio", "low_close_ratio"]

  # drop rows with NaN (should only be first and last rows)
  trainingData.dropna(axis=0, inplace=True)
  
  # loop through data in chunks
  for i in range(start, trainingData.shape[0], step):
    # select chunk of data for training and testing
    train = trainingData.iloc[0:i].copy()
    test = trainingData.iloc[i:(i+step)].copy()
  
    # fit model to data
    model.fit(train[fullTrainingColumns], train["Target"])
  
    # generate predictions from model
    preds = model.predict_proba(test[fullTrainingColumns])[:,1]
    preds = pd.Series(preds, index=test.index)
  
    # classify predictions
    #threshold = 0.5
    #preds[preds > threshold] = 1
    #preds[preds <= threshold] = 0
  
    # combine predictions and test values
    combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)
    predictions.append(combined)
  
  return pd.concat(predictions)

predictions = backtest(trainingData, model, trainingColumns)

# generate an extra field that is the difference between current and previous day close
predictions["difference"] = 0
for i in range(1, predictions.shape[0]):
  current = trainingData["close"].iloc[i]
  previous = trainingData["close"].iloc[i-1]
  predictions["difference"].iloc[i] = current - previous

# scale predictions up
predictions["Predictions"] = predictions["Predictions"]*400-200

# drop target since we don't need to display it
predictions.drop("Target", axis=1, inplace=True)

# convert timestamp column to dates
predictions.index = pd.to_datetime(predictions.index, unit='ms')

predictions.iloc[-100:].plot()
plt.title("Price Predictions")
plt.ylabel('Price $')
plt.xlabel('Date')
plt.tick_params(axis='x', labelrotation=90)
plt.show()