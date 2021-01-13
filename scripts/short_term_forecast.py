import os
import argparse
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from statsmodels.tsa.arima_model import ARIMA as arima
from statsmodels.tsa.statespace.sarimax import SARIMAX as sarima
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")

os.makedirs('forecast_results/AP', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', required=True, help='input dataset')
parser.add_argument('--lag_order', required=False, default=0, help='Number of lag observations(AR)')
parser.add_argument('--diff_degree', required=False, default=0, help='Degree of differencing(I)')
parser.add_argument('--ma_order', required=False, default=1, help='Moving average window size(MA)')
params = parser.parse_args()

lag_order = int(params.lag_order)
diff_degree = int(params.diff_degree)
ma_order = int(params.ma_order)

# load dataframe
df = pd.read_csv(params.filepath)

# separate out validation set
train_data = df.loc[(df['Year'] == 2013)]['Wind Speed'].to_numpy().tolist()
val_data = df.loc[(df['Year'] == 2014) & (df['Month']==1)]['Wind Speed'].to_numpy().tolist()
train_size = len(train_data)
val_size = len(val_data)
print(train_size,val_size)
history = train_data
preds = list()

# forecasting
# for i in range(len(val_data)):
#     model = arima(history, order=(lag_order, diff_degree, ma_order))
#     model_fit = model.fit(disp=False)
#     yhat = model_fit.forecast()[0]
#     preds.append(yhat)
#     history.append(val_data[i])
    # print(f'Predicted = {yhat}; Expected = {val[i]}')

model = sarima(history, order=(lag_order, diff_degree, ma_order), seasonal_order=(0, 1, 1, 24))
model_fit = model.fit(disp=False)
yhat = model_fit.forecast(steps=744)
#preds.append(yhat)
#history.append(val_data[i])

print(model_fit.summary())
rmse = sqrt(mse(val_data, yhat))
print(f"RMSE: {rmse}")

plt.figure()
plt.plot(val_data, color='blue', label='actual')
plt.plot(yhat, color='red', label='prediction')
plt.xlabel('Hour')
plt.ylabel('WindSpeed')
plt.legend(loc="upper left")
plt.savefig('../forecast_results/TN/48hrs_sarima.png')
plt.close()
