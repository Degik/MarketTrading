import yfinance as yf

TICKERS = ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'GOOG', 'AMD']

def get_ticker_data(tickers: list):
    '''
    Obtain the price data for all tickers specified. The outcome is a csv file
    of price data for each ticker in the data folder.
    
    Parameters
    ----------
    tickers : list
        A list of the tickers to download the data for
    '''
        
    data = yf.download(
        tickers = tickers,
        interval = '1d',
        group_by = 'ticker',
        threads = True,
    )
    
    for ticker in tickers:
        
        try:
            df = data.loc[:, ticker.upper()].dropna()
            df.to_csv(f'datasets/{ticker}.csv', index = True)
        except:
            print(f'Ticker {ticker} failed to download.')
            
    return

get_ticker_data(tickers=TICKERS)