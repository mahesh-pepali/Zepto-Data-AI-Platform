import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cleaned",
    "titanic_cleaned.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# Load the tuned complete pipeline

tuned_pipeline_path = os.path.join(
    MODEL_DIR,
    "random_forest_tuned.joblib"
)

tuned_pipeline = joblib.load(
    tuned_pipeline_path
)


# Save tuned pipeline as the final best pipeline

best_pipeline_path = os.path.join(
    MODEL_DIR,
    "best_model_pipeline.joblib"
)

joblib.dump(
    tuned_pipeline,
    best_pipeline_path
)


print("Best-performing pipeline saved.")
print(
    f"Path: {best_pipeline_path}"
)


# Reload the saved pipeline

reloaded_pipeline = joblib.load(
    best_pipeline_path
)


# Load raw cleaned input

df = pd.read_csv(
    DATA_PATH
)


# Remove target columns

X = df.drop(
    columns=["survived", "alive"]
)


# Use one raw sample

raw_sample = X.iloc[
    [0]
].copy()


# Predict using the reloaded complete pipeline

prediction = reloaded_pipeline.predict(
    raw_sample
)


print("\n" + "=" * 50)
print("FINAL PIPELINE RELOAD CHECK")
print("=" * 50)

print(
    "Input sample shape:",
    raw_sample.shape
)

print(
    "Prediction after reload:",
    prediction[0]
)

print(
    "Reload check: SUCCESS"
)

print(
    "\nThe saved pipeline contains both "
    "preprocessing and the tuned Random Forest."
)