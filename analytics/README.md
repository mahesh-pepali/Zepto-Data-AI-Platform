# Analytics Module

This module performs exploratory data analysis, classification modeling, class-imbalance comparison, Random Forest hyperparameter tuning, and a regression side-task using the classic Titanic dataset.

## Dataset and Offline Fallback

The Titanic dataset was loaded once using Seaborn and immediately saved as `analytics/titanic.csv` for offline use.

The original dataset contains 891 rows and 15 columns.

A raw copy is also stored at:

`analytics/data/raw/titanic_raw.csv`

After cleaning, the dataset contains 889 rows and 13 columns with no remaining missing values.

The cleaned dataset is stored at:

`analytics/data/cleaned/titanic_cleaned.csv`

## EDA Findings and Interpretations

### Missing Value Analysis

The `age` column has 19.87% missing values, so median imputation was applied because the missing percentage falls between 5% and 30%.

The `embarked` column has 0.22% missing values, so the affected rows were dropped because the missing percentage is below 5%.

The `deck` column has 77.22% missing values, so it was dropped because imputing such a large proportion of missing values would be unreliable.

The `embark_town` column was also dropped because it duplicates the information represented by `embarked`.

After cleaning, the dataset contains 889 rows and 13 columns with no remaining missing values.

### Outlier Analysis

Using the IQR rule, 65 outliers were identified in `age` and 114 outliers were identified in `fare`.

These observations were retained because the assignment requires identifying and reporting the outliers rather than automatically removing them.

### Fare Distribution

The mean fare is 32.0967, the median is 14.4542, and the mode is 8.05.

Since the mean is greater than the median and the median is greater than the mode, and the skewness is strongly positive at 4.8014, the fare distribution is strongly right-skewed.

The high-fare outliers contribute to the long right tail of the distribution.

### Survival Rate by Sex

Female passengers had a survival rate of 74.04%, while male passengers had a survival rate of 18.89%.

This shows a large difference in survival outcomes between the two sex groups, with female passengers having a substantially higher survival rate.

### Survival Rate by Passenger Class

First-class passengers had a survival rate of 62.62%, compared with 47.28% for second-class and 24.24% for third-class passengers.

Survival therefore decreased as passenger class moved from first to third class, indicating a strong relationship between passenger class and survival.

### Survival Rate by Sex and Passenger Class

Female survival rates were 96.74% in first class, 92.11% in second class, and 50.00% in third class.

Male survival rates were 36.89%, 15.74%, and 13.54% for first, second, and third class respectively.

This shows that both sex and passenger class were important factors associated with survival. Female passengers generally had much higher survival rates, while passengers in higher classes also had better survival outcomes.

### Correlation Analysis

The correlation matrix was restricted to the six required columns:

- `survived`
- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

The columns `adult_male` and `alone` were intentionally excluded from the correlation matrix because they are derived or redundant variables.

The two strongest absolute off-diagonal correlations were:

1. `pclass` and `fare`: -0.5482
2. `sibsp` and `parch`: 0.4145

The negative correlation between `pclass` and `fare` reflects the numerical coding of passenger class, where first class has a lower numerical value than third class while higher-class passengers generally paid higher fares.

The positive correlation between `sibsp` and `parch` indicates that passengers travelling with siblings or spouses were also more likely to travel with parents or children.

The correlation heatmap is saved as:

`analytics/outputs/correlation_heatmap.png`

## Multivariate Chart Interpretations

### 1. Survival Rate by Passenger Class and Sex

The chart shows that female passengers had higher survival rates than male passengers across all three passenger classes.

Survival was also generally higher in first class and lower in third class.

The combination of sex and passenger class therefore provides a clearer picture of survival differences than either variable alone.

Chart:

`analytics/outputs/survival_class_sex.png`

### 2. Age Distribution by Survival and Sex

The chart compares age distributions across survival outcomes for males and females.

Female passengers generally show higher survival proportions, while male passengers have much lower survival across the groups.

The chart also shows that survival differences cannot be explained by age alone and that sex is an important factor.

Chart:

`analytics/outputs/age_survival_sex.png`

### 3. Age, Fare, Survival and Passenger Class

The scatter plot shows the relationship between age and fare while using survival status and passenger class as additional dimensions.

Higher fares are more common among first-class passengers, while lower fares are concentrated among third-class passengers.

The plot also shows that survival was more common among passengers associated with higher fares and higher passenger classes.

Chart:

`analytics/outputs/age_fare_survival_class.png`

### 4. Passenger Class and Survival

The count plot shows the number of passengers who survived and did not survive within each passenger class.

Third class contains the largest number of passengers and also has a much larger number of non-survivors.

First class has a noticeably higher proportion of survivors, reinforcing the relationship between passenger class and survival.

Chart:

`analytics/outputs/class_survival_count.png`

## Exploratory Z-Score Standardization

Z-score standardization was applied only for exploratory analysis to `age` and `fare`.

Before standardization, age had a mean of approximately 29.32 and standard deviation of 12.98, while fare had a mean of approximately 32.10 and standard deviation of 49.70.

After standardization, both variables had means approximately equal to 0 and standard deviations approximately equal to 1.

This exploratory standardization was not used as a replacement for the train-only preprocessing pipeline created during modeling.

## Classification Modeling

### Train/Test Split

The data was divided into training and testing sets using an 80/20 stratified split with `random_state=42`.

Stratification was used so that the proportion of survived and non-survived passengers remained approximately consistent in both the training and testing sets.

The split was performed before preprocessing to prevent information from the test set from influencing the preprocessing steps.

### Preprocessing

The preprocessing pipeline was fitted only on the training data.

Numeric features were processed using median imputation followed by `StandardScaler`.

The numeric features explicitly used by the classification preprocessing pipeline were:

- `pclass`
- `age`
- `sibsp`
- `parch`
- `fare`

Categorical features were processed using most-frequent imputation followed by one-hot encoding.

The categorical features explicitly used by the classification preprocessing pipeline were:

- `sex`
- `embarked`

A `ColumnTransformer` and `Pipeline` were used to keep preprocessing and modeling together and prevent data leakage.

The `survived` target was excluded from the predictors, and `alive` was also excluded because it directly represents the target outcome and would cause target leakage.

The classification models were:

1. Logistic Regression
2. Decision Tree
3. Random Forest

All three classifiers used the same train/test split.

## Classification Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8202 | 0.7812 | 0.7353 | 0.7576 | 0.8179 |

The Random Forest achieved the highest baseline accuracy, recall, and F1-score.

Logistic Regression achieved the highest ROC-AUC among the three baseline classifiers.

The Decision Tree had lower overall performance, particularly for recall and F1-score.

The classification results are saved in:

`analytics/outputs/classification_results.csv`

The ROC curve comparison is saved in:

`analytics/outputs/roc_curve_comparison.png`

The Decision Tree was visualized using scikit-learn's `plot_tree` with labeled feature names and class names, as required. A custom Decision Tree visualization is also retained as an additional supporting artifact.

The required Decision Tree visualization is saved as:

`analytics/outputs/decision_tree.png`

The additional custom visualization is saved as:

`analytics/outputs/decision_tree_custom.png`

## Class Imbalance Comparison

Three Random Forest approaches were compared:

1. Baseline Random Forest
2. Random Forest with `class_weight='balanced'`
3. Random Forest with SMOTE

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.8202 | 0.7812 | 0.7353 | 0.7576 | 0.8179 |
| Class Weight Balanced | 0.8034 | 0.7391 | 0.7500 | 0.7445 | 0.8229 |
| SMOTE | 0.7921 | 0.7460 | 0.6912 | 0.7176 | 0.8250 |

The baseline Random Forest achieved the highest accuracy and F1-score.

Using `class_weight='balanced'` slightly improved recall but reduced accuracy, precision, and F1-score.

SMOTE produced the highest ROC-AUC but had lower accuracy, recall, and F1-score than the baseline model.

The comparison was used to understand the effect of different class-imbalance strategies before hyperparameter tuning.

SMOTE was applied only to the training data to avoid test-set leakage.

The imbalance comparison results are saved in:

`analytics/outputs/imbalance_comparison.csv`

The imbalance comparison chart is saved in:

`analytics/outputs/imbalance_comparison.png`

## Random Forest Hyperparameter Tuning

GridSearchCV was used to tune the Random Forest over:

- `n_estimators`
- `max_depth`
- `max_features`

The Random Forest estimator was created with `oob_score=True`.

The search was evaluated using F1-score.

### Best Parameters

```text
n_estimators = 200
max_depth = 10
max_features = None

The best cross-validation F1-score was:

0.7624

The tuned model achieved an out-of-bag score of:

0.8326

The tuned Random Forest was then evaluated on the held-out test set.
```
## Tuned Random Forest Results

Metric	Tuned Random Forest
Accuracy	0.8315
Precision	0.8065
Recall	0.7353
F1-score	0.7692
ROC-AUC	0.8305
OOB Score	0.8326

The tuned model improved test accuracy and F1-score compared with the baseline Random Forest.

The Random Forest tuning results are saved in:

analytics/outputs/random_forest_tuning_results.csv

The GridSearch results are saved in:

analytics/outputs/grid_search_results.csv

The tuned model confusion matrix is saved in:

analytics/outputs/tuned_random_forest_confusion_matrix.png

## Regression Side-Task

A separate regression task was performed using fare as the continuous target.

The regression pipeline was evaluated using:

MAE
RMSE
R²
Adjusted R²

The results were:

Metric	Result
MAE	18.3735
RMSE	41.2921
R²	0.3609
Adjusted R²	0.2795

The absolute residual correlation was:

0.4115

The residual spread ratio was:

15.6604

The results indicate that the regression model explains part of the variation in passenger fare, while a substantial amount of variation remains unexplained.

The residual analysis is saved in:

analytics/outputs/linear_regression_residuals.png

The regression results are saved in:

analytics/outputs/regression_results.csv

The regression pipeline is saved as:

analytics/models/linear_regression_pipeline.joblib

## Heteroscedasticity Analysis

The residual analysis was used to assess whether the regression errors had a constant spread.

The residual spread ratio of 15.6604 indicates substantial variation in residual spread across the predicted fare range.

The residual analysis therefore suggests the presence of heteroscedasticity rather than a constant residual variance.

The residual plot is stored at:

analytics/outputs/linear_regression_residuals.png

## Final Model Comparison

The final model comparison is stored in:

analytics/outputs/final_model_comparison.csv

Classification and regression metrics are treated as separate metric groups because they measure different prediction tasks.

The classification models predict the binary survived target, while the regression model predicts continuous fare.

The tuned Random Forest achieved:

Accuracy: 0.8315
Precision: 0.8065
Recall: 0.7353
F1-score: 0.7692
ROC-AUC: 0.8305

The regression model achieved:

MAE: 18.3735
RMSE: 41.2921
R²: 0.3609
Adjusted R²: 0.2795

These results are reported separately because classification and regression metrics cannot be directly combined into a single performance scale.

## Saved Models

The trained models are stored in the models directory.

The saved artifacts include:

best_model_pipeline.joblib
decision_tree_baseline.joblib
linear_regression_pipeline.joblib
logistic_regression_baseline.joblib
random_forest_baseline.joblib
random_forest_class_weight.joblib
random_forest_smote.joblib
random_forest_tuned.joblib

The final classification artifact is:

analytics/models/best_model_pipeline.joblib

This artifact contains the complete preprocessing pipeline and final estimator together.

The saved pipeline was reloaded successfully and verified for end-to-end use.

## Project Structure
``` text
analytics/
├── code/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   └── 03_finalize.py
├── data/
│   ├── raw/
│   │   └── titanic_raw.csv
│   └── cleaned/
│       └── titanic_cleaned.csv
├── models/
│   ├── best_model_pipeline.joblib
│   ├── decision_tree_baseline.joblib
│   ├── linear_regression_pipeline.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── random_forest_baseline.joblib
│   ├── random_forest_class_weight.joblib
│   ├── random_forest_smote.joblib
│   └── random_forest_tuned.joblib
├── notebooks/
├── outputs/
├── titanic.csv
├── README.md
└── requirements.txt
``` 
## How to Run

Run the EDA pipeline from the project root:

python analytics/code/01_eda.py

Run the modeling pipeline:

python analytics/code/02_modeling.py

Run the finalization and model verification:

python analytics/code/03_finalize.py

The scripts generate the cleaned dataset, visualizations, evaluation results, and saved model artifacts.

## Requirements

The required Python packages are listed in:

analytics/requirements.txt

The main libraries used by this module include:

Python
Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
Imbalanced-learn
Joblib

## Design Decisions
Dataset Loading

The Titanic dataset is loaded once and saved as analytics/titanic.csv.

This provides an offline fallback so the modeling workflow can run without downloading the dataset again.

Missing Values

Missing values are handled according to the assignment's percentage-based threshold.

age is median-imputed because its missing rate is between 5% and 30%.

The small number of missing embarked values are removed because the missing rate is below 5%.

The highly incomplete deck column is dropped because imputing such a large proportion of missing values would be unreliable.

embark_town is also removed because it duplicates embarked.

Outliers

Outliers in age and fare are identified using the IQR rule.

They are reported rather than automatically removed because the assignment requires identification and interpretation of the outliers.

Correlation Analysis

The correlation matrix is restricted to the six required numeric columns.

The derived boolean columns adult_male and alone are excluded because they are redundant features.

Data Leakage Prevention

The train/test split is performed before preprocessing.

Imputation, scaling, and encoding are fitted only on the training data.

A ColumnTransformer and scikit-learn Pipeline are used to ensure consistent preprocessing.

Classification

Logistic Regression, Decision Tree, and Random Forest are trained using the same train/test split and evaluated using the same metric suite.

Class Imbalance

Baseline Random Forest, class-weighted Random Forest, and SMOTE-based Random Forest are compared.

SMOTE is applied only to the training data.

Hyperparameter Tuning

GridSearchCV is used to tune Random Forest parameters.

Out-of-bag evaluation is enabled using oob_score=True.

Regression

Fare prediction is treated as a separate regression task.

MAE, RMSE, R², and adjusted R² are reported, together with residual analysis for heteroscedasticity.

Final Model Artifact

The complete fitted preprocessing and model pipeline is saved as a single Joblib artifact.

This allows the model to be reloaded and used end-to-end on raw input data.

## Verification

The analytics module was verified by confirming:

- The Titanic dataset was loaded successfully.
- The offline fallback titanic.csv was created.
- The raw dataset contains 891 rows and 15 columns.
- The cleaned dataset contains 889 rows and 13 columns.
- No missing values remain after cleaning.
- Age and fare outliers were calculated using the IQR rule.
- Fare mean, median, mode, and skewness were calculated.
- Survival rates were calculated by sex.
- Survival rates were calculated by passenger class.
- Survival rates were calculated jointly by sex and passenger class.
- The required six-column correlation matrix was generated.
- The two strongest absolute correlations were identified.
- At least four multivariate charts were generated.
- Each major chart has a written interpretation.
- Z-score standardization was explored for age and fare.
- The classification split was stratified.
- Preprocessing was fitted only on training data.
- Logistic Regression, Decision Tree, and Random Forest were evaluated.
- Accuracy, precision, recall, F1-score, and ROC-AUC were reported.
- Baseline, class-weighted, and SMOTE Random Forest models were compared.
- SMOTE was applied only to training data.
- Random Forest hyperparameters were tuned using GridSearchCV.
- The best parameters were recorded.
- The OOB score was recorded.
- The regression task was evaluated using MAE, RMSE, R², and adjusted R².
- Residual analysis was performed for heteroscedasticity.
- Final model comparison results were generated.
- The complete final model pipeline was saved.
- The saved pipeline was reloaded and verified successfully.

The complete analytics workflow is contained inside the /analytics module.


