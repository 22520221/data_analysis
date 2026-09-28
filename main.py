import pandas as pd

df = pd.read_csv("sales.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.describe())

print(df[
    (df["Price"] > 500)
    &
    (df["City"]=="HCM")
])

df["Total"] = df["Price"] * df["Quantity"]

df.to_csv("output.csv", index=False)