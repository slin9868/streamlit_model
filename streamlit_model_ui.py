
'''

Instructions

In this optional assignment, your goal is to build a regression or
classification model on the dataset of your choosing.  After building the model,
save it as a pickle file, and deploy it behind a streamlit application.  This
application should have widgets for user input, and prompt the user to submit
the appropriate input values while returning the results of the models
predictions.  For a submission, you should deploy your application with github
and Streamlit Cloud -- share a link to the final app.

Be creative and have fun, incorporate other elements from the streamlit library.

Streamlit Docs: https://docs.streamlit.io/
Deploy to Streamlit Cloud: https://streamlit.io/cloud

'''

import streamlit as st 
import numpy as np
import pickle

import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime

from sklearn.metrics import mean_squared_error
import pmdarima as pmd
from pmdarima.utils import tsdisplay
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import streamlit as st 
import numpy as np
import pickle

import matplotlib.pyplot as plt
import requests
from io import StringIO
from datetime import datetime

from sklearn.metrics import mean_squared_error
import pmdarima as pmd
from pmdarima.utils import tsdisplay
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.header('HW Model for Retail Sales Prediction')
st.write('Please enter the following information:')

input_info = '''
- **Trend:** {"add", "mul", "additive", "multiplicative", None}
- **Seasonal:** {"add", "mul", "additive", "multiplicative", None}
- **Seasonal Periods:** The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily data with a weekly cycle.
'''

st.write(input_info)

trend_options = ["add", "mul", "additive", "multiplicative", None]
trend = st.selectbox('Trend', trend_options)

seasonal_options = ["add", "mul", "additive", "multiplicative", None]
seasonal = st.selectbox('Seasonal', seasonal_options)

seasonal_period_options = list(range(2,13))
seasonal_periods = st.selectbox('Seasonal Periods', seasonal_period_options)


with open('train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('test.pkl', 'rb') as f:
    test = pickle.load(f)
 
hw = ExponentialSmoothing(
    train['sales']
    , seasonal_periods = seasonal_periods
    , trend = trend
    , seasonal = seasonal
).fit()

df_predictions = test.copy()
df_predictions['hw'] = hw.forecast(len(test))

rmse = mean_squared_error(df_predictions['sales'], df_predictions['hw'], squared=False)

st.write(f'The model inputs are trend = {trend}, seasonal = {seasonal}, seasonal_periods = {seasonal_periods}')
st.write(f'rmse = {rmse}')
st.write(hw.summary())
 
