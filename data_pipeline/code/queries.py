import sqlite3
import pandas as pd

DB_NAME = "data_pipeline/data/books.db"

conn = sqlite3.connect(DB_NAME)

# ============================================================
# QUERY 1: SELECT + WHERE
# Find all books with a 5-star rating
# ============================================================

query1 = """
SELECT title, price_gbp, star_rating
FROM books
WHERE star_rating = 5
"""

print("\n===== QUERY 1 =====")
print(query1)
print("OUTPUT:")
result1 = conn.execute(query1).fetchall()

for row in result1:
    print(row)


# ============================================================
# QUERY 2: ORDER BY + LIMIT
# Find the 5 most expensive books
# ============================================================

query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 5
"""

print("\n===== QUERY 2 =====")
print(query2)
print("OUTPUT:")
result2 = conn.execute(query2).fetchall()

for row in result2:
    print(row)


# ============================================================
# QUERY 3: DISTINCT
# Find all unique categories
# ============================================================

query3 = """
SELECT DISTINCT category_name
FROM categories
"""

print("\n===== QUERY 3 =====")
print(query3)
print("OUTPUT:")
result3 = conn.execute(query3).fetchall()

for row in result3:
    print(row)


# ============================================================
# QUERY 4: BETWEEN
# Find books priced between £20 and £30
# ============================================================

query4 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 30
"""

print("\n===== QUERY 4 =====")
print(query4)
print("OUTPUT:")
result4 = conn.execute(query4).fetchall()

for row in result4:
    print(row)


# ============================================================
# QUERY 5: JOIN
# Display books with their category names
# ============================================================

query5 = """
SELECT
    books.title,
    books.price_gbp,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
"""

print("\n===== QUERY 5 =====")
print(query5)
print("OUTPUT:")
result5 = conn.execute(query5).fetchall()

for row in result5:
    print(row)


# ============================================================
# PANDAS: pd.read_sql
# Read two SQL query results into pandas
# ============================================================

print("\n===== PANDAS READ_SQL: QUERY 1 =====")

df_rating = pd.read_sql(query1, conn)
print(df_rating)


print("\n===== PANDAS READ_SQL: JOIN QUERY =====")

df_join_sql = pd.read_sql(
    """
    SELECT
        books.title,
        books.price_gbp,
        categories.category_name
    FROM books
    JOIN categories
        ON books.category_id = categories.category_id
    """,
    conn
)

print(df_join_sql)


# ============================================================
# PANDAS: pd.merge
# Reproduce the JOIN without SQL
# ============================================================

books_df = pd.read_sql(
    """
    SELECT
        title,
        price_gbp,
        category_id
    FROM books
    """,
    conn
)

categories_df = pd.read_sql(
    """
    SELECT
        category_id,
        category_name
    FROM categories
    """,
    conn
)

df_join_merge = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

df_join_merge = df_join_merge[
    ["title", "price_gbp", "category_name"]
]

print("\n===== PANDAS MERGE RESULT =====")
print(df_join_merge)


# ============================================================
# CHECK EQUIVALENCE
# ============================================================

sql_sorted = df_join_sql.sort_values(
    by=["title", "price_gbp"]
).reset_index(drop=True)

merge_sorted = df_join_merge.sort_values(
    by=["title", "price_gbp"]
).reset_index(drop=True)

print("\n===== JOIN EQUIVALENCE =====")
print(sql_sorted.equals(merge_sorted))


conn.close()