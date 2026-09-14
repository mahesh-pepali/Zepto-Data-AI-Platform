import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Create required directories
os.makedirs("analytics/data/raw", exist_ok=True)
os.makedirs("analytics/data/cleaned", exist_ok=True)
os.makedirs("analytics/outputs", exist_ok=True)


# Load Titanic dataset only once
df = sns.load_dataset("titanic")

print("Titanic dataset loaded successfully.")
print("Shape:", df.shape)


# Save the raw dataset immediately after loading
df.to_csv(
    "analytics/titanic.csv",
    index=False
)

df.to_csv(
    "analytics/data/raw/titanic_raw.csv",
    index=False
)

print("\nRaw dataset saved:")
print("analytics/titanic.csv")
print("analytics/data/raw/titanic_raw.csv")


# Display dataset information
print("\nData information:")
df.info()


# Display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())


# Display dataset shape
print("\nDataset shape:")
print(df.shape)


# Calculate missing values and percentages
missing_count = df.isnull().sum()

missing_percentage = (
    missing_count / len(df)
) * 100

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage
})

print("\nMissing value report:")
print(
    missing_report[
        missing_report["missing_count"] > 0
    ]
)


# Store original missing percentages before cleaning
age_missing_pct = (
    df["age"].isnull().mean() * 100
)

embarked_missing_pct = (
    df["embarked"].isnull().mean() * 100
)

deck_missing_pct = (
    df["deck"].isnull().mean() * 100
)


# Create a copy for cleaning
cleaned_df = df.copy()


# Impute age because its missing percentage is between 5% and 30%
if 5 <= age_missing_pct <= 30:
    age_median = cleaned_df["age"].median()

    cleaned_df["age"] = cleaned_df["age"].fillna(
        age_median
    )

    print(
        f"\nage missing = {age_missing_pct:.2f}% "
        f"-> median imputation"
    )


# Drop rows with missing embarked because missing percentage is below 5%
if 0 < embarked_missing_pct < 5:
    cleaned_df = cleaned_df.dropna(
        subset=["embarked"]
    )

    print(
        f"embarked missing = {embarked_missing_pct:.2f}% "
        f"-> dropped affected rows"
    )


# Drop deck because its original missing percentage is very high
if deck_missing_pct > 30:
    cleaned_df = cleaned_df.drop(
        columns=["deck"]
    )

    print(
        f"deck missing = {deck_missing_pct:.2f}% "
        f"-> dropped column"
    )


# Drop embark_town because it duplicates embarked
if "embark_town" in cleaned_df.columns:
    cleaned_df = cleaned_df.drop(
        columns=["embark_town"]
    )

    print(
        "embark_town -> dropped because it duplicates "
        "the embarked information"
    )


# Display missing values after cleaning
print("\nMissing values after cleaning:")
print(cleaned_df.isnull().sum())


# Save cleaned dataset
cleaned_df.to_csv(
    "analytics/data/cleaned/titanic_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved:")
print(
    "analytics/data/cleaned/titanic_cleaned.csv"
)


# Display final cleaned dataset
print("\nFinal cleaned dataset shape:")
print(cleaned_df.shape)

print("\nFinal columns:")
print(cleaned_df.columns.tolist())

print("\nTotal remaining missing values:")
print(cleaned_df.isnull().sum().sum())

print("\nFirst 5 rows:")
print(cleaned_df.head())


# Calculate IQR outliers for age
age_q1 = cleaned_df["age"].quantile(0.25)
age_q3 = cleaned_df["age"].quantile(0.75)

age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = cleaned_df[
    (cleaned_df["age"] < age_lower) |
    (cleaned_df["age"] > age_upper)
]

print("\nAge IQR outliers:")
print(len(age_outliers))


# Calculate IQR outliers for fare
fare_q1 = cleaned_df["fare"].quantile(0.25)
fare_q3 = cleaned_df["fare"].quantile(0.75)

fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = cleaned_df[
    (cleaned_df["fare"] < fare_lower) |
    (cleaned_df["fare"] > fare_upper)
]

print("\nFare IQR outliers:")
print(len(fare_outliers))


# Calculate fare statistics
fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode()[0]
fare_skewness = cleaned_df["fare"].skew()

print("\nFare statistics:")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)
print("Skewness:", fare_skewness)


# Set Seaborn visualization style
sns.set_theme(style="whitegrid")


# Create age histogram
plt.figure(figsize=(8, 5))

sns.histplot(
    data=cleaned_df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/age_histogram.png"
)

plt.close()


# Create age boxplot
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_df,
    x="age"
)

plt.title("Age Boxplot")
plt.xlabel("Age")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/age_boxplot.png"
)

plt.close()


# Create fare histogram
plt.figure(figsize=(8, 5))

sns.histplot(
    data=cleaned_df,
    x="fare",
    bins=30,
    kde=True
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/fare_histogram.png"
)

plt.close()


# Create fare boxplot
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_df,
    x="fare"
)

plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/fare_boxplot.png"
)

plt.close()


# Calculate survival rate for females using boolean masking
female_mask = cleaned_df["sex"] == "female"

female_survival = cleaned_df[
    female_mask
]["survived"].mean()


# Calculate survival rate for males using boolean masking
male_mask = cleaned_df["sex"] == "male"

male_survival = cleaned_df[
    male_mask
]["survived"].mean()


print("\nSurvival rate by sex:")
print(f"Female: {female_survival:.4f}")
print(f"Male: {male_survival:.4f}")


# Calculate survival rate by passenger class
print("\nSurvival rate by passenger class:")

for pclass in sorted(
    cleaned_df["pclass"].unique()
):

    pclass_mask = (
        cleaned_df["pclass"] == pclass
    )

    survival_rate = cleaned_df[
        pclass_mask
    ]["survived"].mean()

    print(
        f"Class {pclass}: "
        f"{survival_rate:.4f}"
    )


# Calculate survival rate by sex and passenger class
print("\nSurvival rate by sex and passenger class:")

for sex in ["female", "male"]:

    for pclass in [1, 2, 3]:

        mask = (
            (cleaned_df["sex"] == sex) &
            (cleaned_df["pclass"] == pclass)
        )

        survival_rate = cleaned_df[
            mask
        ]["survived"].mean()

        print(
            f"{sex}, Class {pclass}: "
            f"{survival_rate:.4f}"
        )


# Select exactly the six required correlation columns
correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]


# Calculate correlation matrix
correlation_matrix = cleaned_df[
    correlation_columns
].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)


# Find all off-diagonal correlation pairs
correlation_pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        column_1 = correlation_columns[i]
        column_2 = correlation_columns[j]

        correlation_value = (
            correlation_matrix.loc[
                column_1,
                column_2
            ]
        )

        correlation_pairs.append(
            (
                column_1,
                column_2,
                correlation_value,
                abs(correlation_value)
            )
        )


# Sort correlations by absolute value
correlation_pairs = sorted(
    correlation_pairs,
    key=lambda x: x[3],
    reverse=True
)


# Display two strongest correlations
print("\nTwo strongest correlations:")

for pair in correlation_pairs[:2]:

    print(
        f"{pair[0]} and {pair[1]}: "
        f"{pair[2]:.4f}"
    )


# Create correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/correlation_heatmap.png"
)

plt.close()


# Create multivariate chart 1
plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Passenger Class and Sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/survival_class_sex.png"
)

plt.close()


# Create multivariate chart 2
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_df,
    x="survived",
    y="age",
    hue="sex"
)

plt.title(
    "Age Distribution by Survival and Sex"
)

plt.xlabel("Survived")
plt.ylabel("Age")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/age_survival_sex.png"
)

plt.close()


# Create multivariate chart 3
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=cleaned_df,
    x="age",
    y="fare",
    hue="survived",
    style="pclass"
)

plt.title(
    "Age, Fare, Survival and Passenger Class"
)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/age_fare_survival_class.png"
)

plt.close()


# Create multivariate chart 4
plt.figure(figsize=(8, 5))

sns.countplot(
    data=cleaned_df,
    x="pclass",
    hue="survived"
)

plt.title(
    "Passenger Class and Survival"
)

plt.xlabel("Passenger Class")
plt.ylabel("Passenger Count")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/class_survival_count.png"
)

plt.close()


# Create a copy for exploratory z-score standardization
zscore_df = cleaned_df.copy()


# Display values before standardization
print("\nBefore standardization:")

print(
    "Age mean:",
    zscore_df["age"].mean()
)

print(
    "Age std:",
    zscore_df["age"].std()
)

print(
    "Fare mean:",
    zscore_df["fare"].mean()
)

print(
    "Fare std:",
    zscore_df["fare"].std()
)


# Standardize age using z-score
zscore_df["age"] = (
    zscore_df["age"] -
    zscore_df["age"].mean()
) / zscore_df["age"].std()


# Standardize fare using z-score
zscore_df["fare"] = (
    zscore_df["fare"] -
    zscore_df["fare"].mean()
) / zscore_df["fare"].std()


# Display values after standardization
print("\nAfter standardization:")

print(
    "Age mean:",
    zscore_df["age"].mean()
)

print(
    "Age std:",
    zscore_df["age"].std()
)

print(
    "Fare mean:",
    zscore_df["fare"].mean()
)

print(
    "Fare std:",
    zscore_df["fare"].std()
)


# Confirm all charts were saved
print("\nEDA analysis completed successfully.")

print("\nCharts saved in analytics/outputs/:")
print("age_histogram.png")
print("age_boxplot.png")
print("fare_histogram.png")
print("fare_boxplot.png")
print("correlation_heatmap.png")
print("survival_class_sex.png")
print("age_survival_sex.png")
print("age_fare_survival_class.png")
print("class_survival_count.png")