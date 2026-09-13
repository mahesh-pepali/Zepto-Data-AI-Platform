import sqlite3
import pandas as pd

CSV_FILE = "data_pipeline/data/books_cleaned.csv"
DB_FILE = "data_pipeline/data/books.db"

# Load cleaned data
df = pd.read_csv(CSV_FILE)

# Connect to SQLite
conn = sqlite3.connect(DB_FILE)

# Enable foreign keys
conn.execute("PRAGMA foreign_keys = ON")

# Create categories table
conn.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")

# Create books table
conn.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    star_rating INTEGER NOT NULL,
    availability TEXT NOT NULL,
    in_stock BOOLEAN NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")

# Insert categories
for category in df["category"].unique():
    conn.execute(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
        (category,)
    )

# Insert books
for _, row in df.iterrows():

    category_id = conn.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    ).fetchone()[0]

    conn.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            star_rating,
            availability,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["star_rating"],
            row["availability"],
            row["in_stock"],
            category_id
        )
    )

# Save changes
conn.commit()

# Verify counts
book_count = conn.execute(
    "SELECT COUNT(*) FROM books"
).fetchone()[0]

category_count = conn.execute(
    "SELECT COUNT(*) FROM categories"
).fetchone()[0]

print("Data inserted successfully")
print("Books:", book_count)
print("Categories:", category_count)

conn.close()