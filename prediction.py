import os
import torch
import utils
import numpy as np
import pandas as pd
import yfinance as yf

pathname = "netLSTM/datasets/predictions"
ticker = "ISP.MI"

#Download data to predict
def downloadDataToPredict(data:str):
    start, ending = utils.get_20_business_days_back()
    dataset = yf.download(data, start, ending, '1d')
    os.makedirs(pathname, exist_ok=True)
    dataset.to_csv(f"{pathname}/{data}.csv")
    

#Prediction data
downloadDataToPredict(ticker)
data = pd.read_csv(f'{pathname}/{ticker}.csv', sep=",")
print(data)
data = utils.get_lagged_returns(data)
print(data)
#data = utils.get_classification(data)
print(data)
data = data.replace([np.inf, -np.inf], np.nan).dropna()[[col for col in data.columns if 'feat_' in col] + ['classification']]
print(data)