import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

mpg = pd.read_csv("mpg.csv")

# plt.figsize(figsize=(5,4))
sns.scatterplot(data=mpg, x="displ", y="hwy")
plt.show()
plt.clf()

sns.scatterplot(data=mpg, x="displ", y="hwy").set(xlim=[3,6], ylim=[10,30])

sns.scatterplot(data=mpg, x="displ", y="hwy", hue="drv")

# 막대그래프
mpg["drv"].unique()
df_mpg = mpg.groupby("drv", as_index=False).agg(mean_hwy=("hwy", "mean"))
sns.barplot(data=df_mpg, x="drv", y="mean_hwy", hue="drv")

df_mpg2 = mpg.groupby("drv", as_index=False).agg(n=("hwy", "count"))
sns.barplot(data=df_mpg2, x="drv", y="n")

plt.clf()
sns.countplot(data=mpg, x="drv", hue="drv")
plt.show()

sns.countplot(data=mpg, x="drv", hue="drv", order=["4","f","r"])

sns.countplot(data=mpg, x="drv", hue="drv", order=mpg["drv"].value_counts().index)



