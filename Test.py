import pandas as pd
import os as os
from train import train
from forecast import forecast

data_path = "/Users/callum/Documents/GitHub/DATA3888/ARIMA/"

stock1 = pd.read_csv(os.path.join(data_path, "Stock1.csv"), nrows=50000)

model = train(stock1, training_ratio = 0.8)

forecast(model, stock1, training_ratio = 0.8)

params = model.summary()
print(params)


