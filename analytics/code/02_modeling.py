import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned", "titanic_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
    }

    return metrics, y_pred, y_prob


def print_classification_results(name, metrics, y_test, y_pred):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nAccuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1']:.4f}")
    print(f"ROC-AUC: {metrics['ROC_AUC']:.4f}")


def draw_custom_tree(tree_model, feature_names, class_names, output_path):
    tree = tree_model.tree_
    node_positions = {}

    def get_width(node):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            return 1

        return get_width(left) + get_width(right)

    total_width = get_width(0)

    def assign_positions(node, depth, left_edge, right_edge):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            x = left_edge + (right_edge - left_edge) / 2
            node_positions[node] = (x, -depth)
            return

        left_width = get_width(left)
        right_width = get_width(right)

        total = left_width + right_width

        split = left_edge + (
            (right_edge - left_edge) * left_width / total
        )

        node_positions[node] = (
            (left_edge + right_edge) / 2,
            -depth
        )

        assign_positions(
            left,
            depth + 1,
            left_edge,
            split
        )

        assign_positions(
            right,
            depth + 1,
            split,
            right_edge
        )

    assign_positions(
        0,
        0,
        0,
        total_width
    )

    max_depth = tree.max_depth

    fig_width = max(14, total_width * 1.4)
    fig_height = max(8, (max_depth + 1) * 2.2)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    for node in range(tree.node_count):

        if node not in node_positions:
            continue

        x, y = node_positions[node]

        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:

            values = tree.value[node][0]

            predicted_class = int(
                np.argmax(values)
            )

            label = (
                f"Leaf\n"
                f"samples={tree.n_node_samples[node]}\n"
                f"class={class_names[predicted_class]}"
            )

        else:

            feature_index = tree.feature[node]
            threshold = tree.threshold[node]

            feature_name = feature_names[
                feature_index
            ]

            values = tree.value[node][0]

            predicted_class = int(
                np.argmax(values)
            )

            label = (
                f"{feature_name} <= {threshold:.2f}\n"
                f"samples={tree.n_node_samples[node]}\n"
                f"class={class_names[predicted_class]}"
            )

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.5",
                alpha=0.85
            ),
        )

        if left != right:

            parent_x, parent_y = node_positions[node]

            left_x, left_y = node_positions[left]
            right_x, right_y = node_positions[right]

            ax.annotate(
                "",
                xy=(left_x, left_y + 0.08),
                xytext=(parent_x, parent_y - 0.08),
                arrowprops=dict(
                    arrowstyle="->"
                ),
            )

            ax.annotate(
                "",
                xy=(right_x, right_y + 0.08),
                xytext=(parent_x, parent_y - 0.08),
                arrowprops=dict(
                    arrowstyle="->"
                ),
            )

    ax.set_xlim(
        -0.5,
        total_width + 0.5
    )

    ax.set_ylim(
        -max_depth - 1,
        1
    )

    ax.axis("off")

    ax.set_title(
        "Decision Tree Visualization"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# Load cleaned dataset

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Cleaned Titanic dataset loaded.")
print("Shape:", df.shape)


# Classification setup

X = df.drop(
    columns=["survived", "alive"]
)

y = df["survived"]


print("\nTarget class distribution:")
print(y.value_counts())


print("\nTarget class percentage:")
print(
    y.value_counts(normalize=True).mul(100)
)


# Stratified train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# Classification preprocessing

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
])


categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    ),
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    ),
])


# Three classification models

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
}


classification_results = []
roc_data = {}
trained_pipelines = {}


# Train and evaluate classifiers

for name, estimator in models.items():

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            estimator
        ),
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    metrics, y_pred, y_prob = classification_metrics(
        pipeline,
        X_test,
        y_test
    )

    trained_pipelines[name] = pipeline

    roc_data[name] = (
        y_test,
        y_prob
    )

    print_classification_results(
        name,
        metrics,
        y_test,
        y_pred
    )

    classification_results.append({
        "Model": name,
        **metrics
    })


    # Save confusion matrix

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix - {name}"
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    safe_name = name.lower().replace(
        " ",
        "_"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=200
    )

    plt.close()


    # Save complete baseline pipeline

    joblib.dump(
        pipeline,
        os.path.join(
            MODEL_DIR,
            f"{safe_name}_baseline.joblib"
        )
    )


# Classification comparison

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

classification_df = pd.DataFrame(
    classification_results
)

print(
    classification_df.to_string(
        index=False
    )
)

classification_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_results.csv"
    ),
    index=False
)


# ROC curves

plt.figure(figsize=(8, 6))

for name, (
    actual,
    probabilities
) in roc_data.items():

    fpr, tpr, _ = roc_curve(
        actual,
        probabilities
    )

    auc_value = roc_auc_score(
        actual,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_value:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curve_comparison.png"
    ),
    dpi=200
)

plt.close()


# Custom Decision Tree visualization

trained_tree = trained_pipelines[
    "Decision Tree"
]

tree_model = trained_tree.named_steps[
    "model"
]

tree_preprocessor = trained_tree.named_steps[
    "preprocessor"
]

feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

feature_names = [
    name.replace("num__", "")
    .replace("cat__", "")
    for name in feature_names
]

draw_custom_tree(
    tree_model,
    feature_names,
    ["Not Survived", "Survived"],
    os.path.join(
        OUTPUT_DIR,
        "decision_tree.png"
    )
)


# Class imbalance comparison

print("\n" + "=" * 50)
print("CLASS IMBALANCE COMPARISON")
print("=" * 50)


baseline_rf = trained_pipelines[
    "Random Forest"
]

baseline_metrics, _, _ = classification_metrics(
    baseline_rf,
    X_test,
    y_test
)


# Class weight balanced

balanced_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )
    ),
])


balanced_pipeline.fit(
    X_train,
    y_train
)

balanced_metrics, balanced_pred, _ = classification_metrics(
    balanced_pipeline,
    X_test,
    y_test
)


joblib.dump(
    balanced_pipeline,
    os.path.join(
        MODEL_DIR,
        "random_forest_class_weight.joblib"
    )
)


# SMOTE

smote_pipeline = ImbPipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "smote",
        SMOTE(random_state=42)
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    ),
])


smote_pipeline.fit(
    X_train,
    y_train
)

smote_metrics, smote_pred, _ = classification_metrics(
    smote_pipeline,
    X_test,
    y_test
)


joblib.dump(
    smote_pipeline,
    os.path.join(
        MODEL_DIR,
        "random_forest_smote.joblib"
    )
)


imbalance_df = pd.DataFrame([
    {
        "Method": "Baseline",
        **baseline_metrics
    },
    {
        "Method": "Class Weight Balanced",
        **balanced_metrics
    },
    {
        "Method": "SMOTE",
        **smote_metrics
    },
])


print(
    imbalance_df.to_string(
        index=False
    )
)


imbalance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "imbalance_comparison.csv"
    ),
    index=False
)


# Imbalance comparison chart

plt.figure(figsize=(9, 6))

metrics_to_plot = [
    "Precision",
    "Recall",
    "F1"
]

x = np.arange(
    len(imbalance_df)
)

width = 0.25

for i, metric in enumerate(
    metrics_to_plot
):

    plt.bar(
        x + (i - 1) * width,
        imbalance_df[metric],
        width,
        label=metric
    )


plt.xticks(
    x,
    imbalance_df["Method"]
)

plt.ylim(0, 1)

plt.ylabel("Score")

plt.title(
    "Class Imbalance Handling Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "imbalance_comparison.png"
    ),
    dpi=200
)

plt.close()


# Imbalance confusion matrices

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 4)
)

for ax, name, pred in [
    (
        axes[0],
        "Class Weight Balanced",
        balanced_pred
    ),
    (
        axes[1],
        "SMOTE",
        smote_pred
    )
]:

    cm = confusion_matrix(
        y_test,
        pred
    )

    ax.imshow(cm)

    ax.set_title(name)

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "imbalance_confusion_matrices.png"
    ),
    dpi=200
)

plt.close()


# Random Forest GridSearchCV

print("\n" + "=" * 50)
print("RANDOM FOREST GRIDSEARCHCV")
print("=" * 50)

print("\nStarting GridSearchCV...")
print("This may take some time.")


grid_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            oob_score=True,
            bootstrap=True
        )
    ),
])


# Assignment-required parameters

param_grid = {
    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        None
    ]
}


grid_search = GridSearchCV(
    estimator=grid_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)


grid_search.fit(
    X_train,
    y_train
)


tuned_rf = grid_search.best_estimator_


tuned_metrics, tuned_pred, tuned_prob = classification_metrics(
    tuned_rf,
    X_test,
    y_test
)


oob_score = (
    tuned_rf
    .named_steps["model"]
    .oob_score_
)


print("\nBest Parameters:")
print(
    grid_search.best_params_
)


print("\nBest Cross-Validation F1 Score:")
print(
    f"{grid_search.best_score_:.4f}"
)


print("\nOOB Score:")
print(
    f"{oob_score:.4f}"
)


print("\nTuned Random Forest Test Results:")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        tuned_pred
    )
)


print(
    f"\nAccuracy: {tuned_metrics['Accuracy']:.4f}"
)

print(
    f"Precision: {tuned_metrics['Precision']:.4f}"
)

print(
    f"Recall: {tuned_metrics['Recall']:.4f}"
)

print(
    f"F1 Score: {tuned_metrics['F1']:.4f}"
)

print(
    f"ROC-AUC: {tuned_metrics['ROC_AUC']:.4f}"
)


# Baseline vs tuned comparison

baseline_row = classification_df[
    classification_df["Model"] == "Random Forest"
].iloc[0]


comparison_rf_df = pd.DataFrame([
    {
        "Model":
            "Baseline Random Forest",

        "Accuracy":
            baseline_row["Accuracy"],

        "Precision":
            baseline_row["Precision"],

        "Recall":
            baseline_row["Recall"],

        "F1":
            baseline_row["F1"],

        "ROC_AUC":
            baseline_row["ROC_AUC"],

        "OOB_Score":
            np.nan,

        "CV_F1":
            np.nan
    },

    {
        "Model":
            "Tuned Random Forest",

        **tuned_metrics,

        "OOB_Score":
            oob_score,

        "CV_F1":
            grid_search.best_score_
    }
])


print("\n" + "=" * 50)
print("BASELINE VS TUNED RANDOM FOREST")
print("=" * 50)

print(
    comparison_rf_df.to_string(
        index=False
    )
)


comparison_rf_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_tuning_results.csv"
    ),
    index=False
)


# Save GridSearch results

cv_results = pd.DataFrame(
    grid_search.cv_results_
)


cv_columns = [
    "params",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "std_train_score",
    "rank_test_score"
]


cv_results[
    cv_columns
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "grid_search_results.csv"
    ),
    index=False
)


# Tuned RF confusion matrix

cm = confusion_matrix(
    y_test,
    tuned_pred
)

fig, ax = plt.subplots(
    figsize=(5, 4)
)

ax.imshow(cm)

ax.set_title(
    "Tuned Random Forest Confusion Matrix"
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "tuned_random_forest_confusion_matrix.png"
    ),
    dpi=200
)

plt.close()


joblib.dump(
    tuned_rf,
    os.path.join(
        MODEL_DIR,
        "random_forest_tuned.joblib"
    )
)


# Regression side-task

print("\n" + "=" * 50)
print("REGRESSION SIDE-TASK")
print("=" * 50)


# Predict fare using all other available cleaned features

X_reg = df.drop(
    columns=["fare"]
)

y_reg = df["fare"]


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# Automatically identify numeric and categorical columns

reg_numeric_features = (
    X_reg_train
    .select_dtypes(
        include=["number", "bool"]
    )
    .columns
    .tolist()
)


reg_categorical_features = (
    X_reg_train
    .select_dtypes(
        exclude=["number", "bool"]
    )
    .columns
    .tolist()
)


# Regression preprocessing

reg_numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


reg_categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


reg_preprocessor = ColumnTransformer([
    (
        "num",
        reg_numeric_transformer,
        reg_numeric_features
    ),
    (
        "cat",
        reg_categorical_transformer,
        reg_categorical_features
    )
])


# Linear Regression pipeline

regression_pipeline = Pipeline([
    (
        "preprocessor",
        reg_preprocessor
    ),
    (
        "model",
        LinearRegression()
    )
])


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


# Predictions

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


# Regression metrics

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


# Adjusted R2

n = len(y_reg_test)

p = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


if n - p - 1 > 0:

    adjusted_r2 = (
        1
        -
        (
            (1 - r2)
            *
            (n - 1)
            /
            (n - p - 1)
        )
    )

else:

    adjusted_r2 = np.nan


# Residuals

residuals = (
    y_reg_test.to_numpy()
    -
    y_reg_pred
)


abs_residuals = np.abs(
    residuals
)


# Simple heteroscedasticity checks

if (
    len(y_reg_pred) > 1
    and np.std(y_reg_pred) > 0
    and np.std(abs_residuals) > 0
):

    abs_residual_corr = np.corrcoef(
        y_reg_pred,
        abs_residuals
    )[0, 1]

else:

    abs_residual_corr = 0.0


residual_frame = pd.DataFrame({
    "predicted": y_reg_pred,
    "residual": residuals
})


residual_frame[
    "predicted_bin"
] = pd.qcut(
    residual_frame["predicted"],
    q=4,
    duplicates="drop"
)


residual_spread = (
    residual_frame
    .groupby(
        "predicted_bin",
        observed=True
    )["residual"]
    .std()
    .dropna()
)


if len(residual_spread) >= 2:

    spread_ratio = (
        residual_spread.max()
        /
        max(
            residual_spread.min(),
            1e-12
        )
    )

else:

    spread_ratio = 1.0


if (
    abs(abs_residual_corr) >= 0.20
    or spread_ratio >= 1.50
):

    heteroscedasticity_conclusion = (
        "The residual plot suggests heteroscedasticity "
        "because the residual spread changes as predicted "
        "fare increases."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong evidence "
        "of heteroscedasticity; the residual spread is "
        "reasonably stable across predicted fare values."
    )


print(
    f"\nMAE: {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R2: {r2:.4f}"
)

print(
    f"Adjusted R2: {adjusted_r2:.4f}"
)


print(
    "\nResidual absolute-value correlation "
    f"with predictions: {abs_residual_corr:.4f}"
)


print(
    f"Residual spread ratio: {spread_ratio:.4f}"
)


print(
    "\nHeteroscedasticity conclusion:"
)

print(
    heteroscedasticity_conclusion
)


# Save regression results

regression_results = pd.DataFrame([

    {
        "Model":
            "Multivariate Linear Regression",

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2,

        "Adjusted_R2":
            adjusted_r2,

        "Heteroscedasticity":
            heteroscedasticity_conclusion
    }

])


regression_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regression_results.csv"
    ),
    index=False
)


# Residual plot

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Linear Regression Residual Plot"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "linear_regression_residuals.png"
    ),
    dpi=200
)

plt.close()


# Save regression pipeline

joblib.dump(
    regression_pipeline,
    os.path.join(
        MODEL_DIR,
        "linear_regression_pipeline.joblib"
    )
)


# Final model comparison

final_comparison = pd.DataFrame([

    {
        "Model": row["Model"],
        "Model_Type": "Classification",
        "Accuracy": row["Accuracy"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1": row["F1"],
        "ROC_AUC": row["ROC_AUC"],
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Adjusted_R2": np.nan
    }

    for _, row in classification_df.iterrows()

] + [

    {
        "Model":
            "Multivariate Linear Regression",

        "Model_Type":
            "Regression",

        "Accuracy":
            np.nan,

        "Precision":
            np.nan,

        "Recall":
            np.nan,

        "F1":
            np.nan,

        "ROC_AUC":
            np.nan,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2,

        "Adjusted_R2":
            adjusted_r2
    }

])


print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

print(
    final_comparison.to_string(
        index=False
    )
)


final_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_comparison.csv"
    ),
    index=False
)


# Save best complete classification pipeline

best_pipeline = baseline_rf

best_pipeline_path = os.path.join(
    MODEL_DIR,
    "best_model_pipeline.joblib"
)


joblib.dump(
    best_pipeline,
    best_pipeline_path
)


# Reload pipeline and test raw input

reloaded_pipeline = joblib.load(
    best_pipeline_path
)


raw_sample = X_test.iloc[
    [0]
].copy()


reloaded_prediction = (
    reloaded_pipeline.predict(
        raw_sample
    )
)


print("\n" + "=" * 50)
print("BEST PIPELINE SAVE/RELOAD CHECK")
print("=" * 50)

print(
    f"Saved pipeline: {best_pipeline_path}"
)

print(
    "Raw sample prediction after reload:",
    reloaded_prediction[0]
)

print(
    "Pipeline reload check: successful"
)


print(
    "\nClassification modeling and regression completed."
)


print("\nSaved outputs:")

for filename in [

    "classification_results.csv",

    "roc_curve_comparison.png",

    "decision_tree.png",

    "imbalance_comparison.csv",

    "imbalance_comparison.png",

    "random_forest_tuning_results.csv",

    "grid_search_results.csv",

    "tuned_random_forest_confusion_matrix.png",

    "regression_results.csv",

    "linear_regression_residuals.png",

    "final_model_comparison.csv"

]:

    print(filename)


print("\nSaved models:")

for filename in [

    "logistic_regression_baseline.joblib",

    "decision_tree_baseline.joblib",

    "random_forest_baseline.joblib",

    "random_forest_class_weight.joblib",

    "random_forest_smote.joblib",

    "random_forest_tuned.joblib",

    "linear_regression_pipeline.joblib",

    "best_model_pipeline.joblib"

]:

    print(filename)