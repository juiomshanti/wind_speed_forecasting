import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Tamil Nadu/2000.csv")

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=["Minute", "Snow Depth", "Year", "Month", "Day", "Hour"])

# matrix = np.triu(df.corr())
# sns.heatmap(df.corr(), mask=matrix)
sns.heatmap(df.corr())
plt.tight_layout()

plt.savefig("/home/ashryaagr/Desktop/heatmap.png")