# Data Pipeline

This module implements the data collection, cleaning, transformation, SQLite database creation, and SQL/Pandas analysis required for the Zepto Data & AI Platform capstone project.

## Overview

The data pipeline collects book data from Books to Scrape, cleans and transforms the scraped information, stores the structured data in a normalized SQLite database, and performs SQL and Pandas analysis.

The pipeline consists of:

1. Web scraping
2. Data cleaning and transformation
3. CSV generation
4. SQLite database creation
5. SQL queries
6. Equivalent Pandas analysis

## Dataset

The data is scraped from:

https://books.toscrape.com/

The scraper collects the first 3 pages of the website.

- Pages scraped: 3
- Books collected: 60
- Categories: 25

The cleaned dataset is stored at:

`data/books_cleaned.csv`

The SQLite database is stored at:

`data/books.db`

## Extracted Fields

The following fields are collected:

- `title`
- `price_gbp`
- `star_rating`
- `availability`
- `in_stock`
- `category`

An additional derived field is:

- `price_inr`

## Data Cleaning

The following transformations are performed:

- `price_gbp` is converted from text to a floating-point number.
- Star ratings such as `One`, `Two`, `Three`, `Four`, and `Five` are converted to integers from 1 to 5.
- Availability text is retained.
- `in_stock` is converted to a Boolean value.
- Category values are extracted from the book details page.
- Missing values are checked before and after cleaning.
- No missing values were found in the final dataset.
- No rows were dropped because of missing values.
- No artificial values were introduced for missing data.

## Currency Conversion

The assignment requires the fixed conversion rate:

```text
1 GBP = 105.50 INR

Therefore:

price_inr = price_gbp × 105.50

The fixed rate is used so that the results remain reproducible.

## Web Scraping

The scraper is implemented in:

code/scraper.py

The scraper uses:

requests for HTTP requests
BeautifulSoup for HTML parsing

The scraper visits the first three pages and extracts the required book information.

The scraper follows the book detail links to obtain the category information.

The final dataset contains 60 books.

## Database Design

The cleaned data is stored in SQLite.

Database file:

data/books.db

The database uses two normalized tables:

## Categories

Stores unique book categories.

Important fields include:

category_id
category_name

## Books

Stores individual book records.

Important fields include:

book_id
title
price_gbp
price_inr
star_rating
availability
in_stock
category_id

The category_id field connects the books table to the categories table through a foreign-key relationship.

This avoids repeating the same category name for every book record.

## Database Creation

Database creation is implemented in:

code/database.py

The script:

Creates the SQLite database.
Creates the categories table.
Creates the books table.
Inserts unique categories.
Inserts the cleaned book records.
Establishes the category relationship using the foreign key.

The resulting database contains:

60 books
25 categories

## SQL Analysis

SQL queries are implemented in:

code/queries.py

The analysis contains at least five SQL queries covering the required SQL functionality.

The queries include operations such as:

Filtering records
Sorting records
Aggregation
Grouping
Category-based analysis
Joining the books and categories tables

The executed query outputs are stored in:

outputs/query_outputs.txt

## Pandas Analysis

The SQL analysis is also reproduced using Pandas.

Pandas is used to read the SQLite data and perform equivalent operations.

The implementation uses:

pd.read_sql()

and:

pd.merge()

The Pandas results are compared with the corresponding SQL results.

The SQL and Pandas join results match.

## Query Output

The generated SQL query results are stored in:

outputs/query_outputs.txt

This file contains the printed output of the executed SQL analysis.

## Project Structure

data_pipeline/
├── code/
│   ├── scraper.py
│   ├── database.py
│   └── queries.py
├── data/
│   ├── books_cleaned.csv
│   └── books.db
├── outputs/
│   └── query_outputs.txt
├── README.md
└── requirements.txt

## How to Run

From the project root, activate the project virtual environment first.

Run the scraper:

python data_pipeline/code/scraper.py

This collects the book data and creates the cleaned CSV.

Create the SQLite database:

python data_pipeline/code/database.py

Run the SQL and Pandas analysis:

python data_pipeline/code/queries.py

The query results are written to:

data_pipeline/outputs/query_outputs.txt

## Requirements

The required Python packages are listed in:

requirements.txt

The main libraries used are:

requests
beautifulsoup4
pandas

SQLite is provided through Python's standard library.

## Design Decisions

Scraping:

The first three pages of Books to Scrape were used because the assignment requires a controlled dataset size while still providing enough records for database analysis.

Cleaning:

The scraped text values are converted into appropriate numeric and Boolean types before being stored.

Currency Conversion:

A fixed conversion rate of 105.50 INR per GBP is used instead of a live exchange rate so that the results are reproducible.

Category Normalization:

Categories are stored separately in a categories table and referenced from the books table using a foreign key.

This reduces duplicate category values and provides a normalized relational structure.

## SQL and Pandas

The required analysis is implemented in both SQL and Pandas.

pd.read_sql() is used to retrieve SQL results, while pd.merge() is used to reproduce the relational join in Pandas.

The outputs are compared to verify consistency.

## Verification

The data pipeline was verified by confirming:

3 pages were scraped.
60 books were collected.
25 categories were identified.
The cleaned CSV was generated successfully.
No missing values remained in the cleaned dataset.
The SQLite database was created successfully.
The database contains 60 book records.
The database contains 25 categories.
The books and categories tables are connected through a foreign key.
At least five SQL queries were executed.
SQL query outputs were saved.
Equivalent Pandas operations were performed.
SQL and Pandas join results match.

The complete data pipeline is contained inside the /data_pipeline module.

