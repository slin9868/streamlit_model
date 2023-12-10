
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
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.header('HW Model for Retail Sales Prediction')
st.write('Please enter the following information:')

# Define a dictionary to store user input
user_input = {}

# Define a boolean flag to track if button was clicked
is_button_clicked = False

input_info = '''
- **Trend:** {"add", "mul", "additive", "multiplicative", None}
- **Seasonal:** {"add", "mul", "additive", "multiplicative", None}
- **Seasonal Periods:** The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily data with a weekly cycle.
'''

st.write(input_info)

# Use a placeholder to display the output after button click
output_placeholder = st.empty()

# Create the "Find out Best Param Set" button
button = st.button("Find out Best Param Set")

if button:
  is_button_clicked = True

# Show output only after button click
if is_button_clicked:
  with open('params_df.pkl', 'rb') as f:
    params_df = pickle.load(f)
  # Display the output using the placeholder
  output_placeholder.write(params_df.sort_values(by='aic', ascending=True))

trend_options = ["add", "mul", "additive", "multiplicative", None]
trend = st.selectbox('Trend', trend_options, key="trend")
user_input["trend"] = trend

seasonal_options = ["add", "mul", "additive", "multiplicative", None]
seasonal = st.selectbox('Seasonal', seasonal_options, key="seasonal")
user_input["seasonal"] = seasonal

seasonal_period_options = list(range(2,13))
seasonal_periods = st.selectbox('Seasonal Periods', seasonal_period_options, key="seasonal_periods")
user_input["seasonal_periods"] = seasonal_periods

# Load the pickled data
with open('train.pkl', 'rb') as f:
  train = pickle.load(f)

with open('test.pkl', 'rb') as f:
  test = pickle.load(f)

# Train the Holt-Winters model using user-selected parameters
hw = ExponentialSmoothing(
  train['sales'],
  seasonal_periods = seasonal_periods,
  trend = trend,
  seasonal = seasonal
).fit()

# Generate predictions and calculate RMSE
df_predictions = test.copy()
df_predictions['hw'] = hw.forecast(len(test))
rmse = mean_squared_error(df_predictions['sales'], df_predictions['hw'], squared=False)

# Display relevant information and charts
st.write(f'RMSE and Summary with model inputs : trend = {trend}, seasonal = {seasonal}, seasonal_periods = {seasonal_periods}')
st.write(f'RMSE = {rmse}')
st.write(hw.summary())

st.write("\n")
st.write(f'Chart with model inputs: trend = {trend}, seasonal = {seasonal}, seasonal_periods = {seasonal_periods}')

plt.figure(figsize=(12, 8))
plt.plot(train['sales'], label='Train')
plt.plot(test['sales'], label='Test')
plt.plot(df_predictions['hw'], label="Holt-Winters'")
plt.legend()
plt.title('Forecasts with Holt-Winters');

st.pyplot(plt)
 
