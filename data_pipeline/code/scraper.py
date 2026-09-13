import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Convert star rating text to numbers
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


# Convert price text to float
def clean_price(price):
    price = price.replace("Â£", "")
    price = price.replace("£", "")
    return float(price)


# Base URL for pagination
base_url = "https://books.toscrape.com/catalogue/page-{}.html"

book_data = []


# Scrape first 3 pages
for page in range(1, 4):

    url = base_url.format(page)

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.find_all("article", class_="product_pod")

    print(f"Page {page}: {len(books)} books")


    # Extract data from each book
    for book in books:

        # Extract title
        title = book.h3.a["title"]


        # Extract price
        price_text = book.find(
            "p",
            class_="price_color"
        ).get_text(strip=True)

        price = clean_price(price_text)


        # Extract rating
        rating_text = book.find(
            "p",
            class_="star-rating"
        )["class"][1]

        rating = rating_map.get(rating_text)


        # Extract availability
        availability = book.find(
            "p",
            class_="instock availability"
        ).get_text(strip=True)

        in_stock = availability == "In stock"


        # Get book page URL
        book_link = book.h3.a["href"]

        book_url = urljoin(url, book_link)


        # Open the individual book page
        book_response = requests.get(book_url)

        book_soup = BeautifulSoup(
            book_response.text,
            "html.parser"
        )
        

     
        # Extract category
        category = "Unknown"

        category_link = book_soup.select_one(
        "ul.breadcrumb li:nth-of-type(3) a"
        )

        if category_link:
          category = category_link.get_text(strip=True)
        # Store book information
        book_data.append({
        "title": title,
        "price_gbp": price,
        "star_rating": rating,
        "availability": availability,
        "in_stock": in_stock,
        "category": category
        })


# Create DataFrame
df = pd.DataFrame(book_data)


# Convert GBP to INR using the required fixed exchange rate
GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

# Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())


# Remove rows with missing required values
required_columns = [
    "title",
    "price_gbp",
    "star_rating",
    "availability",
    "in_stock",
    "category"
]

df = df.dropna(subset=required_columns)


# Check for missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Check final number of rows
print("\nFinal row count:", len(df))


# Check price conversion
print("\nPrice conversion:")
print(
    df[
        ["title", "price_gbp", "price_inr"]
    ].head()
)


# Check total number of books
print("\nTotal books:", len(df))


# Display categories
print("\nCategories:")
print(df["category"].unique())


# Display category counts
print("\nCategory counts:")
print(df["category"].value_counts())

# Save cleaned dataset
df.to_csv(
    "data_pipeline/data/books_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved to books_cleaned.csv")

# Display first five rows
print("\nFirst 5 rows:")
print(df.head())


# Display data types
print("\nData types:")
print(df.dtypes)