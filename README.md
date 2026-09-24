# Zepto Data & AI Platform

End-to-End AI/ML Capstone Project

This repository implements an end-to-end data and AI platform for Zepto. The project combines web scraping and data engineering, exploratory data analysis and machine learning, and a retrieval-based customer support assistant.

The project is divided into three independent modules:

1. Data Pipeline
2. Analytics
3. Support Assistant

Each module contains its own code, data, outputs, requirements, and README documentation.

## Project Overview

The Zepto Data & AI Platform demonstrates a complete workflow from data collection to machine learning and AI-powered customer support.

The platform consists of:

```text
Web Data
   |
   v
Data Pipeline
   |
   +--> Scraping
   +--> Cleaning
   +--> SQLite Database
   +--> SQL/Pandas Analysis
   |
   v
Analytics
   |
   +--> EDA
   +--> Data Cleaning
   +--> Visualization
   +--> Classification
   +--> Class Imbalance
   +--> Hyperparameter Tuning
   +--> Regression
   |
   v
Support Assistant
   |
   +--> Policy Corpus
   +--> Sentence Embeddings
   +--> ChromaDB
   +--> LangGraph
   +--> Retrieval
   +--> Response Generation
   +--> FastAPI
   +--> Docker
```
## Project Objectives

The main objectives of the project are:

- Collect structured data from a public web source.
- Clean and transform the collected data.
- Store structured data in a normalized SQLite database.
- Perform SQL and Pandas analysis.
- Perform exploratory data analysis.
- Build and evaluate machine learning models.
- Compare class-imbalance strategies.
- Tune a Random Forest model.
- Perform a regression side-task.
- Build a policy-based AI support assistant.
- Implement semantic retrieval using embeddings.
- Store embeddings in ChromaDB.
- Implement a LangGraph workflow.
- Expose the support assistant through FastAPI.
- Containerize the support assistant using Docker.
- Maintain the complete project in a single GitHub repository.
## Repository Structure
``` text
zepto-data-ai-platform/
│
├── README.md
├── .gitignore
│
├── data_pipeline/
│   ├── code/
│   │   ├── scraper.py
│   │   ├── database.py
│   │   └── queries.py
│   │
│   ├── data/
│   │   ├── books_cleaned.csv
│   │   └── books.db
│   │
│   ├── outputs/
│   │   └── query_outputs.txt
│   │
│   ├── README.md
│   └── requirements.txt
│
├── analytics/
│   ├── code/
│   │   ├── 01_eda.py
│   │   ├── 02_modeling.py
│   │   └── 03_finalize.py
│   │
│   ├── data/
│   │   ├── raw/
│   │   │   └── titanic_raw.csv
│   │   └── cleaned/
│   │       └── titanic_cleaned.csv
│   │
│   ├── models/
│   │   ├── best_model_pipeline.joblib
│   │   ├── decision_tree_baseline.joblib
│   │   ├── linear_regression_pipeline.joblib
│   │   ├── logistic_regression_baseline.joblib
│   │   ├── random_forest_baseline.joblib
│   │   ├── random_forest_class_weight.joblib
│   │   ├── random_forest_smote.joblib
│   │   └── random_forest_tuned.joblib
│   │
│   ├── notebooks/
│   ├── outputs/
│   ├── titanic.csv
│   ├── README.md
│   └── requirements.txt
│
└── support_assistant/
    ├── code/
    │   ├── main.py
    │   ├── ingest.py
    │   ├── graph.py
    │   ├── prompt.py
    │   ├── test_graph.py
    │   └── test_retrieval.py
    │
    ├── corpus/
    │   ├── doc_01_delivery_policy.txt
    │   ├── doc_02_returns_refunds.txt
    │   ├── doc_03_membership_tiers.txt
    │   ├── doc_04_order_tracking.txt
    │   ├── doc_05_order_cancellation.txt
    │   ├── doc_06_damaged_missing.txt
    │   ├── doc_07_gift_cards.txt
    │   └── doc_08_support_hours.txt
    │
    ├── data/
    │   └── chroma_db/
    │
    ├── Dockerfile
    ├── README.md
    └── requirements.txt
```
## Module 1: Data Pipeline

The Data Pipeline module collects book information from Books to Scrape, cleans and transforms the collected data, creates a normalized SQLite database, and performs SQL and Pandas analysis.

Main Components
``` text
Web Source
    |
    v
scraper.py
    |
    v
Data Cleaning
    |
    v
books_cleaned.csv
    |
    v
database.py
    |
    v
books.db
    |
    v
queries.py
    |
    +--> SQL Analysis
    |
    +--> Pandas Analysis
```
## Data Collection

The scraper collects book information from the Books to Scrape website.

The pipeline collects data from the first three pages, producing:

- 60 books
- 25 categories

The collected fields include:

- Title
- Price in GBP
- Price converted to INR
- Star rating
- Availability
- Stock status
- Category

The GBP-to-INR conversion uses a fixed conversion value of:

GBP_TO_INR = 105.50

## Database

The cleaned data is stored in SQLite.

The database uses normalized tables for:

- Categories
- Books

The category relationship is represented using a foreign key.

The resulting database is:

data_pipeline/data/books.db

## SQL and Pandas Analysis

Five SQL queries are implemented and their results are stored in:

data_pipeline/outputs/query_outputs.txt

Equivalent analysis is also performed using Pandas.

Running the Data Pipeline

From the project root:

python data_pipeline/code/scraper.py

Create/update the database:

python data_pipeline/code/database.py

Run the analysis:

python data_pipeline/code/queries.py

For detailed documentation, see:

data_pipeline/README.md

## Module 2: Analytics

The Analytics module performs exploratory data analysis and machine learning using the Titanic dataset.

## The module includes:

- Dataset loading
- Offline dataset fallback
- Data cleaning
- Missing-value analysis
- Outlier analysis
- Distribution analysis
- Correlation analysis
- Multivariate visualization
- Z-score standardization
- Classification
- Class imbalance comparison
- Random Forest tuning
- Regression
- Residual analysis
- Model persistence

## Dataset

The Titanic dataset contains:

891 rows
15 columns

After cleaning:

889 rows
13 columns

The cleaned dataset contains no remaining missing values.

An offline copy is stored at:

analytics/titanic.csv

## Exploratory Data Analysis

The EDA includes analysis of:

- Missing values
- Outliers
- Fare distribution
- Survival by sex
- Survival by passenger class
- Survival by sex and passenger class
- Correlations
- Multivariate relationships

The analysis generates multiple visualizations in:

analytics/outputs/

## Classification

Three baseline classification models were evaluated:

Logistic Regression
Decision Tree
Random Forest

The classification metrics include:

Accuracy
Precision
Recall
F1-score
ROC-AUC

Baseline results:
``` text
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.8090	0.7833	0.6912	0.7344	0.8610
Decision Tree	0.7640	0.7600	0.5588	0.6441	0.8374
Random Forest	0.8202	0.7812	0.7353	0.7576	0.8179
```
## Class Imbalance

Three Random Forest approaches were compared:

- Baseline Random Forest
- Class-weighted Random Forest
- SMOTE Random Forest
``` text
Method	Accuracy	Precision	Recall	F1	ROC-AUC
Baseline	0.8202	0.7812	0.7353	0.7576	0.8179
Class Weight Balanced	0.8034	0.7391	0.7500	0.7445	0.8229
SMOTE	0.7921	0.7460	0.6912	0.7176	0.8250
```
## Random Forest Tuning

GridSearchCV was used to tune:

n_estimators
max_depth
max_features

Best parameters:

n_estimators = 200
max_depth = 10
max_features = None

Best cross-validation F1-score:

0.7624

OOB score:

0.8326

Test results for the tuned model:

Metric	Result
Accuracy	0.8315
Precision	0.8065
Recall	0.7353
F1-score	0.7692
ROC-AUC	0.8305
OOB Score	0.8326
Regression

A separate regression task was performed using fare as the target.

Results:

Metric	Result
MAE	18.3735
RMSE	41.2921
R²	0.3609
Adjusted R²	0.2795

Residual analysis was also performed to investigate heteroscedasticity.

The residual spread ratio was:

15.6604

The results suggest non-constant residual variance.

Saved Models

The final model pipeline is stored at:

analytics/models/best_model_pipeline.joblib

The saved pipeline contains the preprocessing and final estimator together.

The saved model was reloaded and verified successfully.

Running the Analytics Module

From the project root:

python analytics/code/01_eda.py

Run modeling:

python analytics/code/02_modeling.py

Run finalization and verification:

python analytics/code/03_finalize.py

For detailed documentation, see:

analytics/README.md
Module 3: Support Assistant

The Support Assistant module implements an AI-powered policy support system using semantic retrieval, embeddings, ChromaDB, LangGraph, Pydantic, FastAPI, and Docker.

The assistant is designed to answer questions using a predefined Zepto policy corpus.

Architecture
Customer Query
      |
      v
FastAPI /ask
      |
      v
LangGraph
      |
      v
classify_intent
      |
      +-----------------------------+
      |                             |
      | policy_question             | general_question
      v                             v
retrieve_and_answer           direct_answer
      |                             |
      v                             |
Query Embedding                    |
      |                             |
      v                             |
ChromaDB Retrieval                 |
      |                             |
      v                             |
Top 3 Policy Chunks                |
      |                             |
      +-------------+---------------+
                    |
                    v
             Pydantic Response
                    |
                    v
             JSON API Response
Policy Corpus

The assistant contains eight policy documents:

Delivery Policy
Returns and Refunds
Membership Tiers
Order Tracking
Order Cancellation
Damaged or Missing Items
Gift Cards
Support Hours

The documents are stored in:

support_assistant/corpus/
Embeddings

The policy documents are embedded using:

all-MiniLM-L6-v2

from Sentence Transformers.

The embeddings are stored in a persistent ChromaDB collection named:

zepto_policy

The ChromaDB database is stored at:

support_assistant/data/chroma_db/
Retrieval

For policy questions:

The user query is converted into an embedding.
ChromaDB performs semantic similarity search.
The top three most similar policy documents are retrieved.
The retrieved context is passed to the response generation stage.
The response contains the retrieved document IDs as sources.

Cosine similarity is used through normalized embeddings and ChromaDB retrieval.

Intent Classification

The classifier uses the required keyword heuristic.

A query is classified as a policy_question if it contains one of the following keywords:

delivery
return
refund
membership
tracking
cancel
gift card
support hours

Otherwise it is classified as:

general_question
LangGraph

The LangGraph workflow contains three required nodes:

classify_intent
retrieve_and_answer
direct_answer

The classifier determines which branch is executed.

Policy questions are routed to:

retrieve_and_answer

General questions are routed to:

direct_answer
Prompt Design

The retrieval prompt follows a structured template containing:

Role
Context
Task
Format
Length

It also contains:

An explicit negative constraint
A few-shot example

The prompt instructs the assistant to use only the retrieved policy context and avoid inventing policy information.

Mock LLM Mode

The default execution mode is:

MOCK_LLM=1

This allows the complete support assistant to run offline without an API key.

For a policy query, the mock response is based on the top retrieved policy chunk.

For a general question, the mock response is:

I can only answer questions about Zepto policies right now.
Response Schema

The final response is validated using Pydantic.

The schema contains:

{
  "answer": "string",
  "sources": [],
  "confidence": 1.0
}

The fields are:

answer — generated response
sources — retrieved policy document IDs
confidence — value between 0 and 1
FastAPI

The support assistant exposes:

POST /ask

Request:

{
  "query": "How does order tracking work?"
}

Example response:

{
  "answer": "Based on the retrieved context: Customers can track their order using the live rider map from the packed stage until delivery. The Track Order screen shows the current order status and estimated time of arrival. The ETA is updated a",
  "sources": [
    "doc_04_order_tracking",
    "doc_01_delivery_policy",
    "doc_05_order_cancellation"
  ],
  "confidence": 1.0
}

A general question such as:

{
  "query": "What is the capital of India?"
}

returns:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
Running the Support Assistant Locally

First build the ChromaDB collection:

python support_assistant/code/ingest.py

Start FastAPI:

uvicorn support_assistant.code.main:app --reload --port 7860

The API will be available at:

http://127.0.0.1:7860

Swagger documentation is available at:

http://127.0.0.1:7860/docs
Testing the Graph

Run:

python -m support_assistant.code.test_graph

This tests both:

Policy-question routing
General-question routing
Testing Retrieval

Run:

python -m support_assistant.code.test_retrieval

This verifies that policy queries retrieve relevant documents from ChromaDB.

Docker

The Support Assistant includes a Dockerfile for local containerized execution.

Build the image:

docker build -t zepto-support-assistant .

Run the container:

docker run --rm -p 7860:7860 zepto-support-assistant

The Docker container:

Installs the required dependencies.
Builds the policy embedding collection.
Starts the FastAPI application.
Exposes the service on port 7860.

The container uses:

MOCK_LLM=1

by default, so the baseline application does not require an external API key.

Requirements

Each module maintains its own requirements file.

Data Pipeline
data_pipeline/requirements.txt
Analytics
analytics/requirements.txt
Support Assistant
support_assistant/requirements.txt

The project primarily uses:

Python
Requests
BeautifulSoup
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn
Joblib
Sentence Transformers
ChromaDB
LangGraph
Pydantic
FastAPI
Uvicorn
LangChain OpenAI
Docker
Installation

Create and activate a Python virtual environment.

On Windows:

python -m venv capstone

Activate it:

.\capstone\Scripts\Activate.ps1

Install the requirements required for the module you want to run.

For Data Pipeline:

pip install -r data_pipeline/requirements.txt

For Analytics:

pip install -r analytics/requirements.txt

For Support Assistant:

pip install -r support_assistant/requirements.txt
End-to-End Execution

The three modules can be executed independently.

Step 1: Data Pipeline
python data_pipeline/code/scraper.py
python data_pipeline/code/database.py
python data_pipeline/code/queries.py

This produces the cleaned book dataset, SQLite database, and SQL/Pandas analysis.

Step 2: Analytics
python analytics/code/01_eda.py
python analytics/code/02_modeling.py
python analytics/code/03_finalize.py

This produces the EDA outputs, model evaluation results, visualizations, and saved models.

Step 3: Support Assistant
python support_assistant/code/ingest.py
uvicorn support_assistant.code.main:app --reload --port 7860

Then open:

http://127.0.0.1:7860/docs
Design Decisions
Data Pipeline

The pipeline uses a lightweight requests and BeautifulSoup approach for web scraping.

SQLite was selected because the project requires a local relational database and the dataset is small enough for a local database.

The database is normalized into categories and books to avoid unnecessary duplication.

SQL results are compared with equivalent Pandas operations.

Analytics

An offline copy of the dataset is stored so the project can be reproduced without repeatedly downloading the source dataset.

Preprocessing is implemented inside scikit-learn pipelines to reduce the risk of data leakage.

Multiple classification models are compared instead of relying on a single algorithm.

Class imbalance strategies are explicitly evaluated.

GridSearchCV is used for systematic Random Forest tuning.

The final preprocessing and model are saved together as a Joblib pipeline.

Support Assistant

The support assistant uses a fixed policy corpus so that responses can be grounded in defined information.

Sentence Transformers provides lightweight local embeddings.

ChromaDB provides persistent vector storage and semantic retrieval.

LangGraph provides explicit workflow routing between classification, retrieval, and direct-answer paths.

Pydantic validates the final response structure.

FastAPI provides a lightweight API interface.

Docker provides a reproducible deployment environment.

MOCK_LLM=1 is used as the default baseline so the application can run without paid services or external API credentials.

Reproducibility

The project is designed so that each module can be run independently from the project root.

The main reproducibility features are:

Fixed random states for machine learning experiments.
Local dataset fallbacks.
Saved cleaned datasets.
Saved model artifacts.
Persistent ChromaDB storage.
Module-specific requirements files.
Docker support for the Support Assistant.
README instructions for each module.
Git Workflow

The project was developed using feature branches and merged into the main branch.

The repository history includes separate development branches for the modules.

The workflow followed the pattern:

main
 |
 +-- feature/data-pipeline
 |       |
 |       +-- commits
 |       |
 |       +-- merge into main
 |
 +-- feature/analytics
 |       |
 |       +-- commits
 |       |
 |       +-- merge into main
 |
 +-- feature/support-assistant
         |
         +-- commits
         |
         +-- merge into main

The Git history therefore records the development of the major project modules separately.

Module Documentation

Detailed documentation is available inside each module.

Data Pipeline
data_pipeline/README.md

This covers:

Web scraping
Data cleaning
SQLite database
SQL queries
Pandas analysis
Outputs
Analytics
analytics/README.md

This covers:

EDA
Data cleaning
Visualizations
Classification
Class imbalance
Random Forest tuning
Regression
Model artifacts
Verification
Support Assistant
support_assistant/README.md

This covers:

Policy corpus
Embeddings
ChromaDB
Prompt design
LangGraph
Retrieval
Mock LLM
FastAPI
Docker
Verification
Verification Checklist

The complete project was verified across the three modules.

Data Pipeline
 Web scraping implemented
 Book data collected
 Categories extracted
 Data cleaning implemented
 Price conversion implemented
 Clean CSV generated
 SQLite database created
 Normalized tables created
 SQL queries implemented
 Pandas equivalents implemented
 Query outputs generated
Analytics
 Titanic dataset loaded
 Offline dataset fallback created
 Missing values analyzed
 Missing values handled
 Outliers identified
 Fare distribution analyzed
 Survival analysis completed
 Correlation analysis completed
 Multivariate charts generated
 Chart interpretations documented
 Z-score standardization performed
 Train/test split performed
 Leakage prevention applied
 Logistic Regression trained
 Decision Tree trained
 Random Forest trained
 Classification metrics calculated
 Class imbalance strategies compared
 Random Forest tuned
 OOB evaluation performed
 Regression side-task completed
 Residual analysis completed
 Final model saved
 Saved model reloaded and verified
Support Assistant
 Eight policy documents created
 Sentence Transformer embeddings implemented
 ChromaDB collection created
 Top-3 retrieval implemented
 Prompt template implemented
 Negative constraint included
 Few-shot example included
 LangGraph StateGraph implemented
 TypedDict state implemented
 Three graph nodes implemented
 Conditional routing implemented
 Mock LLM mode implemented
 Pydantic response validation implemented
 Policy-question testing completed
 General-question testing completed
 FastAPI endpoint implemented
 API tested locally
 Docker image built successfully
 Docker container tested successfully
Project Outputs

The main generated outputs are stored inside their respective module directories.

data_pipeline/outputs/
analytics/outputs/
support_assistant/data/chroma_db/

Model artifacts are stored in:

analytics/models/

Datasets are stored in:

data_pipeline/data/
analytics/data/
analytics/titanic.csv
Conclusion

The Zepto Data & AI Platform combines three stages of an end-to-end AI/ML workflow:

Data Collection
      |
      v
Data Engineering
      |
      v
Data Analysis
      |
      v
Machine Learning
      |
      v
Model Persistence
      |
      v
AI Retrieval System
      |
      v
API Deployment
      |
      v
Docker Deployment

The project demonstrates practical implementation of data engineering, exploratory analysis, machine learning, retrieval-augmented support workflows, API development, and containerization within a single repository.


**This is the README that should go at the project root:**

```text
zepto-data-ai-platform/
└── README.md

Your three module READMEs remain separately inside:

data_pipeline/README.md
analytics/README.md
support_assistant/README.md

So you will have 4 README files total: one overall RE
