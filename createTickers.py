import yfinance as yf
import utils

TICKERS = ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMD', 'ISP.MI', 'LMT', 'INTC', 'NKE', 'RTX']

def get_ticker_data(tickers: list):
    
    #start, end = utils.get_60_days_back()
    #print(start)
    #print(end)
    
    data = yf.download(
        tickers = tickers,
        interval = '1d',
        group_by = 'ticker',
        threads = True,
    )
    
    for ticker in tickers:
        
        try:
            df = data.loc[:, ticker.upper()].dropna()
            df.to_csv(f'netLSTM/datasets/{ticker}.csv', index = True)
        except:
            print(f'Ticker {ticker} failed to download.')
            
    return

get_ticker_data(tickers=TICKERS)