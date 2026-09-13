# Data Pipeline

## Overview

This module implements the data collection, cleaning, transformation,
SQLite database creation, and SQL/Pandas analysis required for the
Zepto Data & AI Platform capstone project.

## Dataset

The data is scraped from:

https://books.toscrape.com/

The scraper collects the first 3 pages of books.

- Pages scraped: 3
- Books collected: 60
- Categories: 25

## Extracted Fields

The following fields are collected:

- title
- price_gbp
- star_rating
- availability
- in_stock
- category

An additional derived field is:

- price_inr

## Data Cleaning

The following transformations are performed:

- `price_gbp` is converted from text to float.
- Star ratings such as One, Two, Three, Four, and Five are converted
  to integers from 1 to 5.
- Availability text is retained.
- `in_stock` is converted to a Boolean value.
- Category values are extracted from the book details page.
- Missing values are checked before and after cleaning.
- No missing values were found in the final dataset, so no rows were dropped.
- No artificial values were introduced for missing data.

## Currency Conversion

The assignment requires the fixed conversion rate:

1 GBP = 105.50 INR

Therefore:

```text
price_inr = price_gbp × 105.50