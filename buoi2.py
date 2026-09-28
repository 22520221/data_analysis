import requests
from bs4 import BeautifulSoup

url = "https://quotes.toscrape.com/"
response = requests.get(url)

print("Status Code:", response.status_code)

soup = BeautifulSoup(response.text, "lxml")

quotes = soup.find_all("div", class_="quote")

for quote in quotes:
    text = quote.find("span", class_="text").text
    author = quote.find("small", class_="author").text

    print("=" * 30)
    print(text)
    print("Author:", author)

print("Title:")
print(soup.title.text)

print("\nH1:")
print(soup.find("h1").text)

print("\nCác thẻ p:")

paragraphs = soup.find_all("p")

for p in paragraphs:
    print(p.text)