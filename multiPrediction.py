import os
import torch
import utils
import NetMarket
import numpy as np
import pandas as pd
import yfinance as yf

pathname = "netLSTM/datasets/predictions"
TICKERS = ['RTX', 'NVDA', 'MSFT', 'LMT', 'INTC', 'TSLA', 'AAPL', 'GOOG', 'AMD', 'NKE', 'BA']

#Download data to predict
def downloadDataToPredict(data:str):
    start, ending = utils.get_20_business_days_back()
    dataset = yf.download(data, start, ending, '1d')
    os.makedirs(pathname, exist_ok=True)
    dataset.to_csv(f"{pathname}/{data}.csv")
    


for ticker in TICKERS:
    #Prediction data
    downloadDataToPredict(ticker)
    data = pd.read_csv(f'{pathname}/{ticker}.csv', sep=",")
    data = utils.get_lagged_returns(data)
    data = utils.get_classification(data)
    #print(data)
    data = (data.dropna().reset_index(drop = True))
    #print(data)

    data = utils.reshape_x(
            data[[col for col in data.columns if 'feat_' in col] + ['classification']]
            .values[:, :-1]
        )

    #print(data)

    model = torch.load('models/test7/model.pth')
    data = torch.tensor(data)
    data = data.to("cuda:0")
    model = model.to("cuda:0")
    #print(data)

    model.eval()
    with torch.no_grad():
        outputs = model(data)
        
    outputs = torch.max(outputs, 1)
    prediction = outputs[1]
    
    
    if prediction == 0:
        print(f'{ticker} sarà in discesa')
    elif prediction == 1:
        print(f'{ticker} sarà stabile')
    elif prediction == 2:
        print(f'{ticker} sarà in salita')