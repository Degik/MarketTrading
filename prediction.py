import os
import torch
import utils
import NetMarket
import numpy as np
import pandas as pd
import yfinance as yf

pathname = "netLSTM/datasets/predictions"
ticker = "NFLX"

#Download data to predict
def downloadDataToPredict(data:str):
    start, ending = utils.get_20_business_days_back()
    dataset = yf.download(data, start, ending, '1d')
    os.makedirs(pathname, exist_ok=True)
    dataset.to_csv(f"{pathname}/{data}.csv")
    

#Prediction data
downloadDataToPredict(ticker)
data = pd.read_csv(f'{pathname}/{ticker}.csv', sep=",")
data = utils.get_lagged_returns(data)
data = utils.get_classification(data)
print(data)
data = (data.dropna().reset_index(drop = True))
print(data)

data = utils.reshape_x(
        data[[col for col in data.columns if 'feat_' in col] + ['classification']]
        .values[:, :-1]
    )

print(data)

model = torch.load('models/test7/model.pth')
data = torch.tensor(data)
data = data.to("cuda:0")
model = model.to("cuda:0")
print(data)

model.eval()
with torch.no_grad():
    outputs = model(data)
    
outputs = torch.max(outputs, 1)
element = outputs[1][0]
print(element)