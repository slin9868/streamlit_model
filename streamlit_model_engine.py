
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests
from io import StringIO
from sklearn.metrics import mean_squared_error

from datetime import datetime

import pmdarima as pmd
from pmdarima.utils import tsdisplay
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 

import pickle

api_key = 'DA4A3SSW41OW5XTG' # try diff key if 25 limit is reached
api_key = 'OK8HS4M0725ST9C7'
api_key = 'Z3R969RBY5731IFS'

base_url = 'https://www.alphavantage.co/query'
def get_time_series():
    req = requests.get(
        base_url,
        params={
            "function": "RETAIL_SALES",
            "apikey": api_key,
            "datatype": "csv"
        }
    )
    df = pd.read_csv(StringIO(req.text))

    rate_limit_error_msg = "Our standard API rate limit is 25 requests per day."
    if rate_limit_error_msg in req.text:
        print("Rate limit exceeded. Please try again later.")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df.rename(columns={'value': 'sales'}, inplace=True)
    
    return df

retail_sales_data = get_time_series()

# data for the last decade
# start_date = pd.Timestamp.now() - pd.DateOffset(years=10)
# retail_sales_data = retail_sales_data[retail_sales_data['timestamp'] >= start_date]

# data since 2015
retail_sales_data = retail_sales_data[retail_sales_data.index >= '2015-01-01']

# retail_sales_data.head()

train = retail_sales_data.loc[:'2020-01']
test = retail_sales_data.loc['2020-02-01':]

hw = ExponentialSmoothing(
    train['sales']
    # , seasonal_periods=12
    # , trend='add'
    # , seasonal='mul'
).fit()

df_predictions = test.copy()
df_predictions['hw'] = hw.forecast(len(test))

train.to_pickle('train.pkl')
test.to_pickle('test.pkl')
df_predictions.to_pickle('df_predictions.pkl')

print(mean_squared_error(df_predictions['sales'], df_predictions['hw'], squared=False))
