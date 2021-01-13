import pandas as pd
from statsmodels.tsa.stattools import adfuller


df = pd.read_csv("data/Tamil Nadu/TN.csv")
data2 = df.loc[(df['Year'] == 2014)]['Wind Speed']
data = df.loc[(df['Year'] == 2014) & (df['Month']==10)]['Wind Speed']

X = data
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

