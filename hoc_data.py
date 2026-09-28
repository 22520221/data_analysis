import pandas as pd

df = pd.read_csv("sales.csv")

df.head()

df.tail()

df.shape(5, 5)

df.columns

df.info()

df.describe()

df["Price"]

df[df["Price"] > 1000]

df[df["City"] == "HCM"]

df["Total"] = df["Price"] * df["Quantity"]

df["Total"].sum()

df.to_csv("output.csv", index=False)


