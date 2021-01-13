import pandas as pd

df = pd.read_csv("data/Tamil Nadu/TN.csv")
df['week'] = 0

week = 1
j = 1
for i in range(len(df)):
    if df['Month'][i] == 1 & df['Day'][i] == 1 & df['Hour'][i]== 0:
        print(week)
        week = 1
        j = 1
    df['week'][i] = week
    if j % 7 == 0:
        week += 1
    if df['Hour'][i] == 23:
        j += 1

print(df)
# df[["Hour", "Wind Speed"]].groupby("Hour").mean().to_csv("data/Tamil Nadu/hourly.csv")
# df[["Day", "Wind Speed"]].groupby("Day").mean().to_csv("data/Tamil Nadu/daily.csv")
