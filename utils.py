import datetime
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

FEAT_LENGTH = 20
FEAT_COLS = ['Open', 'Low', 'High', 'Close']

def importDatasetX(file_name:str) -> pd.DataFrame:
    dataset = []
    try:
        dataset = pd.read_csv(file_name, header=None, sep=",", skiprows=1)
    except Exception as e:
        print("Error | Can not read dataset cup for take input")
        exit(1)
    dataset = dataset.iloc[:, :-3] # Remove 3 columns
    columns_name = ['DATE'] + [f'X{i}' for i in range(1,4)]
    dataset.columns = columns_name
    dataset = dataset.drop('DATE', axis=1) #Drop column
    return dataset.round(4)

def importDatasetY(file_name:str) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(file_name, header=None, sep=",", skiprows=1)
    except Exception as e:
        print("Error | Can not read dataset cup for take output")
        exit(1)
    columns_list = ['DATE', 'Y1']
    indexes = [0, 4] # take the first and fourth column indexe
    dataset = dataset.iloc[:, indexes]
    dataset.columns = columns_list
    dataset = dataset.drop('DATE', axis=1) #Drop column
    return dataset.round(4)



def get_20_business_days_back():
    #today = "2023-12-22"
    today = datetime.date.today()
    business_day_20_back = pd.to_datetime(today) - BDay(22)
    return business_day_20_back.strftime('%Y-%m-%d'), today

def get_60_days_back():
    today = datetime.date.today()
    date_60_days_back = today - datetime.timedelta(days=55)
    return date_60_days_back.strftime('%Y-%m-%d'), today

def time_series(df: pd.DataFrame,
                col: str,
                name: str) -> pd.DataFrame:
    '''
    Form the lagged columns for this feature
    '''
    return df.assign(**{
        f'{name}_t-{lag}': col.shift(lag)
        for lag in range(0, FEAT_LENGTH)
    })
    
def get_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    For each of the feature cols, find the returns and then form the lagged
    time-series as new columns
    '''
    for col in FEAT_COLS:
        return_col = df[col]/df[col].shift(1)-1
        df = time_series(df, return_col, f'feat_{col}_ret')
        
    return df


def get_classification(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Get the classifications for the LSTM network, which are as follows:
        0 = The 20 period SMA is below the low of the day
        1 = The 20 period SMA is between the low and high of the day
        2 = The 20 period SMA is above the high of the day
    '''
    
    df['ma'] = df['Close'].rolling(20).mean()
    
    conditions = [
        df['ma'] <= df['Low'],
        (df['ma'] < df['High']) & (df['ma'] > df['Low']),
        df['ma'] >= df['High'],
    ]
    
    df['classification'] = np.select(
        condlist = conditions,
        choicelist = [0, 1, 2],
    )
    
    return df

def reshape_x(x: np.array) -> np.array:
    
    # Calculate the number of features we have in the nn (assumes all features
    # are of the same length)
    num_feats = x.shape[1]//FEAT_LENGTH
    
    # Initialise the new x array with the correct size
    x_reshaped = np.zeros((x.shape[0], FEAT_LENGTH, num_feats))
    
    # Populate this array through iteration
    for n in range(0, num_feats):
        x_reshaped[:, :, n] = x[:, n*FEAT_LENGTH:(n+1)*FEAT_LENGTH]
    
    return x_reshaped