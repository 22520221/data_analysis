import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://quotes.toscrape.com/"

response = requests.get(url)

print(response.status_code)

soup = BeautifulSoup(response.text, "lxml")

quotes = soup.find_all("div", class_="quote")

print(len(quotes))  

quotes_list = []
authors = []

for quote in quotes:
    text = quote.find("span", class_="text").text
    author = quote.find("small", class_="author").text

    quotes_list.append(text)
    authors.append(author)

df = pd.DataFrame({
    "Quote": quotes_list,
    "Author": authors
})
print(df.head())
print(df.info())
print(df.describe(include="all"))
print(df["Author"].value_counts())
df.to_csv("quotes.csv", index=False)

