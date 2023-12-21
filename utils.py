import numpy as np
import pandas as pd

def importDatasetX(file_name:str) -> pd.DataFrame:
    dataset = []
    try:
        dataset = pd.read_csv(file_name, header=None, dtype=float)
    except Exception as e:
        print("Error | Can not read dataset cup for take input")
        exit(1)
    dataset = dataset.iloc[:, :-3] # Remove 3 columns
    columns_name = ['DATE'] + [f'X{i}' for i in range(1,3)]
    dataset.columns = columns_name
    dataset.set_index('DATE', inplace=True)
    return dataset.round(4)

def importDatasetY(file_name:str) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(file_name, header=None, dtype=float)
    except Exception as e:
        print("Error | Can not read dataset cup for take output")
        exit(1)
    columns_list = ['DATE', 'Y1']
    indexes = [0, 4] # take the first and fourth column indexe
    dataset = dataset.iloc[:, indexes]
    dataset.columns = columns_list
    dataset.set_index('DATE', inplace=True)
    return dataset.round(4)