import requests
from bs4 import BeautifulSoup
import pandas as pd


# =========================
# BƯỚC 1: Tạo list chứa HTML
# =========================

all_quotes = []

for page in range(1, 4):

    url = f"https://quotes.toscrape.com/page/{page}/"

    response = requests.get(url)

    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "lxml")

    quotes = soup.find_all("div", class_="quote")

    print("Page:", page)
    print("So quote:", len(quotes))

    for quote in quotes:
        all_quotes.append(quote)


print("Tong so quote:", len(all_quotes))


# =========================
# BƯỚC 2: Tách Quote và Author
# =========================

quotes_list = []
authors = []

for quote in all_quotes:

    text = quote.find("span", class_="text").text
    quotes_list.append(text)

    author = quote.find("small", class_="author").text
    authors.append(author)


print("So quote:", len(quotes_list))
print("So author:", len(authors))


# =========================
# BƯỚC 3: Tạo DataFrame
# =========================

df = pd.DataFrame({
    "Quote": quotes_list,
    "Author": authors
})


# =========================
# BƯỚC 4: Xem dữ liệu
# =========================

print(df.head())

df.to_csv("quotes_30.csv", index=False)