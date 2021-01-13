import pandas as pd 

df = pd.read_csv('../data/Andhra Pradesh/2014.csv')
df = df.loc[(df["Month"] >= 8) & (df["Month"] <= 10)]

new_dict = {
    "Hour": df["Hour"],
    "Wind Speed": df['Wind Speed']
}
new_df = pd.DataFrame(new_dict)
new_df.to_csv('../data/Andhra Pradesh/2014_hourly.csv', index=False)

