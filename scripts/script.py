import argparse
from math import sqrt
import numpy as np
from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.arima_model import ARIMA as arima
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', required=True, help='input dataset')
parser.add_argument('--lag_order', required=False, default=1, help='Number of lag observations(AR)')
parser.add_argument('--diff_degree', required=False, default=1, help='Degree of differencing(I)')
parser.add_argument('--ma_order', required=False, default=1, help='Moving average window size(MA)')
params = parser.parse_args()

lag_order = int(params.lag_order)
diff_degree = int(params.diff_degree)
ma_order = int(params.ma_order)

savepath = 'forecast_results/MP/'

# load dataframe
df = pd.read_csv(params.filepath)
df = df['Wind Speed']
# print(df.head())
x = df.to_numpy().tolist()

# ACF and PACF plots
# plt.figure()
# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# plt.subplot(211)
# plt.xlabel("Lags")
# plot_acf(df, ax=plt.gca(), lags=70)
# plt.subplot(212)
# plt.xlabel("Lags")
# plot_pacf(df, ax=plt.gca(), lags=70)
# plt.savefig(savepath+'w_stationary_acf_pacf.png')
# plt.close()

# AD Fuller Test
# print(30*"-")
# print("AD Fuller Test")
# result = adfuller(x)
# print(f"ADF Statistic: {result[0]}")
# print(f"p-value: {result[1]}")
# print("Critical Values:")
# for k, v in result[4].items():
#     print(f"\t{k}: {v}")
# print(30*"-")

# KPSS Test
# print(30*"-")
# print("KPSS Test")
# result = kpss(x)
# print(f"KPSS Statistic: {result[0]}")
# print(f"p-value: {result[1]}")
# print("Critical Values:")
# for k, v in result[3].items():
#     print(f"\t{k}: {v}")
# print(30*"-")


# separate out validation set
train_size = int(len(x)*0.8) # 80-20 train-val split
train, val = x[:train_size], x[train_size:]
print(len(train), len(val))
history = train
preds = list()

# forecasting
for i in range(len(val)):
    model = arima(history, order=(lag_order, diff_degree, ma_order))
    model_fit = model.fit(disp=False)
    yhat = model_fit.forecast()[0]
    preds.append(yhat)
    history.append(val[i])
    # print(f'Predicted = {yhat}; Expected = {val[i]}')

print(model_fit.summary())
rmse = sqrt(mse(val, preds))
print(f"RMSE: {rmse}")

plt.figure()
plt.plot(val, color='blue', label='actual')
plt.plot(preds, color='red', label='prediction')
plt.legend(loc="upper left")
plt.xlabel('Weeks')
plt.ylabel('Wind Speed(m/s)')
plt.savefig(savepath+'w_arima.png')
plt.close()
