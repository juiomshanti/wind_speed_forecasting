import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt




########
# df = pd.read_csv("data/Tamil Nadu/weekly.csv")
# dates = pd.date_range(start='1/1/2018', end='31/12/2018',periods=53)
# df.drop(['week'],axis=1,inplace=True)

# df.set_index(dates,inplace=True)

# print(df.head)
# result = seasonal_decompose(df, model='additive')
# result.plot()
# pyplot.show()

################


df = pd.read_csv("data/Tamil Nadu/TN.csv")
dates1 = pd.date_range(start='8/1/2014',freq='H',periods=744)
dates2 = pd.date_range(start='9/1/2014',freq='H',periods=720)
dates3 = pd.date_range(start='10/1/2014',freq='H',periods=744)


data1 = df.loc[(df['Year'] == 2014) & (df['Month']==8)]['Wind Speed']
data2 = df.loc[(df['Year'] == 2014) & (df['Month']==9)]['Wind Speed']
data3 = df.loc[(df['Year'] == 2014) & (df['Month']==10)]['Wind Speed']


req1 = pd.DataFrame(list(data1),index=dates1,columns=['wind_speed'])
req2 = pd.DataFrame(list(data2),index=dates2,columns=['wind_speed'])
req3 = pd.DataFrame(list(data3),index=dates3,columns=['wind_speed'])

res1 = seasonal_decompose(req1, model='multiplicative')
res2 = seasonal_decompose(req2, model='multiplicative')
res3 = seasonal_decompose(req3, model='multiplicative')
res3.plot()
plt.show()
plt.savefig('aug_2014.png')

# def plotseasonal(res, axes):
#     res.observed.plot(ax=axes[0], legend=False)
#     axes[0].set_ylabel('Observed')
#     res.trend.plot(ax=axes[1], legend=False)
#     axes[1].set_ylabel('Trend')
#     res.seasonal.plot(ax=axes[2], legend=False)
#     axes[2].set_ylabel('Seasonal')
#     res.resid.plot(ax=axes[3], legend=False)
#     axes[3].set_ylabel('Residual')

# fig, axes = plt.subplots(ncols=3, nrows=4, sharex=True, figsize=(12,5))
# plotseasonal(res1, axes[:,0])
# plotseasonal(res2, axes[:,1])
# plotseasonal(res3, axes[:,2])
# plt.tight_layout()
# plt.show()

###############################

# df = pd.read_csv("data/Tamil Nadu/TN.csv")
# dates = pd.date_range(start='1/1/2014',freq='H',periods=8760)
# data = df.loc[df['Year'] == 2014]['Wind Speed']
# #print(data.head)
# print(dates)
# req = pd.DataFrame(list(data),index=dates,columns=['wind_speed'])
# print(req.shape)


# result = seasonal_decompose(req, model='multiplicative')
# result.plot()
# plt.show()

###############################

