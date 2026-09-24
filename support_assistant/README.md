# Zepto Support Assistant

This module implements a policy-grounded Zepto customer support assistant using document embeddings, ChromaDB, LangGraph, Pydantic structured responses, and FastAPI.

The module supports a fully offline mock-LLM mode by default. A real LLM path is also available as an optional extension.

## 1. Module Overview

The Support Assistant answers questions about Zepto policies using a small policy document corpus.

The pipeline contains:

1. Policy document ingestion
2. Text embedding
3. ChromaDB vector storage
4. Intent classification
5. Similarity-based retrieval
6. LangGraph workflow routing
7. Structured response validation
8. FastAPI API serving

The default mode is fully offline and deterministic.

## 2. Project Structure

```text
support_assistant/
├── code/
│   ├── main.py
│   ├── ingest.py
│   ├── graph.py
│   ├── prompt.py
│   ├── test_graph.py
│   ├── test_retrieval.py
│   └── __init__.py
├── corpus/
│   ├── doc_01_delivery_policy.txt
│   ├── doc_02_returns_refunds.txt
│   ├── doc_03_membership_tiers.txt
│   ├── doc_04_order_tracking.txt
│   ├── doc_05_order_cancellation.txt
│   ├── doc_06_damaged_missing.txt
│   ├── doc_07_gift_cards.txt
│   └── doc_08_support_hours.txt
├── data/
│   └── chroma_db/
├── Dockerfile
├── README.md
└── requirements.txt
```
## 3. Policy Corpus

The assistant uses eight policy documents.

Delivery Policy

doc_01_delivery_policy.txt

Contains delivery time, delivery charges, priority delivery, and serviceable pin-code information.

Returns and Refunds

doc_02_returns_refunds.txt

Contains return windows, refund processing, non-returnable items, and free pickup information.

Membership Tiers

doc_03_membership_tiers.txt

Contains information about Zepto Basic, Zepto Pass, Zepto Pass+, pricing, benefits, and cancellation.

Order Tracking

doc_04_order_tracking.txt

Contains order tracking, live rider map, order status, ETA, and support conditions.

Order Cancellation

doc_05_order_cancellation.txt

Contains cancellation timing, Packed-stage restrictions, and automatic cancellation refunds.

Damaged or Missing Items

doc_06_damaged_missing.txt

Contains reporting requirements, replacement/refund information, and photo requirements.

Gift Cards

doc_07_gift_cards.txt

Contains denominations, delivery method, validity, payment combinations, and cash-exchange restrictions.

Support Hours

doc_08_support_hours.txt

Contains chat support availability, email support, response times, and phone-support information.

## 4. Architecture

The overall architecture is:
``` text
Policy Documents
      |
      v
Document Ingestion
      |
      v
Sentence Transformer Embeddings
      |
      v
ChromaDB Vector Collection
      |
      v
User Query
      |
      v
LangGraph
      |
      +----------------------+
      |                      |
      v                      v
classify_intent       general_question
      |
      +----------------------+
      |
      v
policy_question
      |
      v
retrieve_and_answer
      |
      v
Top-3 Similar Documents
      |
      v
Prompt Construction
      |
      v
Mock/Real LLM
      |
      v
Pydantic Validation
      |
      v
Structured JSON Response
```
The FastAPI application exposes the final workflow through the /ask endpoint.

## 5. Embedding and ChromaDB

The embedding model used by the module is:

all-MiniLM-L6-v2

The model is provided through sentence-transformers.

Each policy document is converted into an embedding and stored in ChromaDB.

The ChromaDB collection is:

zepto_policy

The persistent database is stored at:

data/chroma_db/

Each document is stored with:

Document ID
Document text
Source filename
Embedding

The retrieval system returns the top three most similar documents for policy questions.

## 6. Ingestion

Document ingestion is implemented in:

code/ingest.py

The ingestion process:

Reads all policy .txt files from the corpus directory.
Validates that eight policy documents are present.
Loads the all-MiniLM-L6-v2 embedding model.
Generates normalized embeddings.
Creates or loads the ChromaDB collection.
Stores the documents, IDs, metadata, and embeddings.

Run ingestion from the project root:

python -m support_assistant.code.ingest

The expected result is:

Collection: zepto_policy
Documents stored: 8
## 7. Prompt Design

Prompt construction is implemented in:

code/prompt.py

The prompt follows the required structure:

Role
Context
Task
Format
Length

It also contains an explicit negative constraint.

The negative constraint instructs the assistant not to use information that is not present in the supplied policy context.

The prompt also contains a few-shot example demonstrating how a policy question should be answered from retrieved context.

The prompt builder is:

build_prompt(query, context)

The generated prompt contains the user question and retrieved policy context.

## 8. LangGraph Workflow

The LangGraph workflow is implemented in:

code/graph.py

The graph uses a typed state:

class SupportState(TypedDict, total=False):

The workflow contains three required nodes:

classify_intent
retrieve_and_answer
direct_answer

The graph starts with:

classify_intent

The classifier determines whether the query is a policy question or a general question.

The conditional routing is:
``` text
policy_question
        |
        v
retrieve_and_answer

general_question
        |
        v
direct_answer
```
Both paths then terminate at the LangGraph END node.

## 9. Mock LLM and Optional Real LLM

The module uses the environment variable:

MOCK_LLM

The default behavior is:

MOCK_LLM=1

or leaving the variable unset.

In mock mode, no external LLM API is required.

Policy Question

For a policy question:

The query is embedded.
ChromaDB retrieves the top three documents.
The highest-ranked document is selected.
A short snippet is returned in the mock answer.
Retrieved document IDs are returned as sources.
Confidence is set to 1.0.

The mock response follows:

Based on the retrieved context: <top retrieved document snippet>
General Question

For a general question, the mock response is:

I can only answer questions about Zepto policies right now.

No retrieval is performed and the sources list is empty.

Real LLM Mode

The optional real LLM path can be enabled using:

MOCK_LLM=0

An OPENAI_API_KEY is required for this path.

The optional model can be configured with:

OPENAI_MODEL

The default configured model name is:

gpt-4o-mini

The real LLM response is parsed and validated against the Pydantic response schema.

If schema validation fails, the implementation retries the generation up to three total attempts.

## 10. Pydantic Response Schema

The final response is validated using Pydantic.

The response contains exactly these fields:

{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}

The schema is:

class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float

The confidence value is constrained to:

0.0 <= confidence <= 1.0

Mock responses are deterministic.

For policy questions, the sources contain the retrieved document IDs.

For general questions, the sources list is empty.

## 11. FastAPI

The API is implemented in:

code/main.py

The application provides:

GET /

and:

POST /ask

The request format is:

{
  "query": "How does order tracking work?"
}

The response is validated using the AskResponse Pydantic model.

Run the API from the project root:

uvicorn support_assistant.code.main:app --reload --port 7860

The API will be available at:

http://127.0.0.1:7860

FastAPI's interactive documentation is available at:

http://127.0.0.1:7860/docs

## 12. Example API Calls
Example 1: Policy Question

Request:

$body = @{ query = "How does order tracking work?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body $body

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

The query is classified as a policy question and routed through retrieval.

Example 2: General Question

Request:

$body = @{ query = "What is the capital of India?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body $body

Example response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

The query is classified as a general question and does not use policy retrieval.

## 13. Testing

Two test scripts are included.

Retrieval Test
code/test_retrieval.py

This verifies that a policy query retrieves relevant documents from ChromaDB.

For example:

How can I track my order?

The retrieval system identifies:

doc_04_order_tracking

as the top matching document.

LangGraph Test
code/test_graph.py

The test verifies both routing paths.

Run it with:

python -m support_assistant.code.test_graph

The test includes:

How does order tracking work?

and:

What is the capital of India?

The first query is routed to:
``` text
classify_intent
        |
        v
retrieve_and_answer
```
The second query is routed to:
``` text
classify_intent
        |
        v
direct_answer
```
## 14. Docker

The module includes a Dockerfile:

Dockerfile

The Docker image uses Python 3.11 and installs CPU-only PyTorch to avoid unnecessary CUDA dependencies.

The container automatically performs policy ingestion before starting FastAPI.

Build the image from the project root:

docker build -t zepto-support-assistant .

Run the container:

docker run --rm -p 7860:7860 zepto-support-assistant

The container starts the application on:

0.0.0.0:7860

The API can then be accessed locally at:

http://127.0.0.1:7860

The Docker environment uses:

MOCK_LLM=1

so the container works without an API key.

## 15. Requirements

The required packages are listed in:

requirements.txt

The main technologies used are:

Python
FastAPI
Pydantic
ChromaDB
Sentence Transformers
LangGraph
LangChain OpenAI integration
Uvicorn

The embedding model is:

all-MiniLM-L6-v2

The baseline mock mode does not require an external LLM API key.

## 16. Design Decisions

Offline-First Design

The default MOCK_LLM mode is fully deterministic and does not require an external LLM service.

This makes the application reproducible and suitable for offline grading.

Vector Retrieval

ChromaDB is used for persistent local vector storage.

Policy documents are embedded using all-MiniLM-L6-v2 and retrieved using similarity search.

Top-3 Retrieval

The retrieval node requests the top three most similar policy documents.

This provides enough context while keeping the retrieved context compact.

Intent Classification

The required keyword-based classifier is used for deterministic routing.

Policy keywords include:

delivery
return
refund
membership
tracking
cancel
gift card
support hours

Queries containing these keywords are classified as:

policy_question

All other queries are classified as:

general_question
Structured Output

Pydantic is used to validate the final response.

This ensures that the API consistently returns:

answer
sources
confidence
Prompt Grounding

The policy response prompt explicitly instructs the generation step to use only the retrieved policy context and not invent unsupported information.

## 17. End-to-End Flow

The complete system flow is:
``` text
Policy TXT Files
       |
       v
ingest.py
       |
       v
Sentence Transformer
all-MiniLM-L6-v2
       |
       v
ChromaDB
zepto_policy
       |
       |
       v
User POST /ask
       |
       v
classify_intent
       |
       +----------------------+
       |                      |
       v                      v
policy_question        general_question
       |                      |
       v                      v
retrieve_and_answer    direct_answer
       |
       v
Top-3 ChromaDB Results
       |
       v
Prompt Construction
       |
       v
Mock/Real LLM
       |
       v
Pydantic Validation
       |
       v
FastAPI JSON Response
``` 
## 18. Verification

The Support Assistant was verified by confirming:

- Eight policy documents are present.
- All eight documents are embedded successfully.
- ChromaDB collection zepto_policy contains eight documents.
- all-MiniLM-L6-v2 is used for embeddings.
- Policy queries retrieve relevant documents.
- The top three similar documents are returned by retrieval.
- The prompt contains Role, Context, Task, Format, and Length sections.
- The prompt contains an explicit negative constraint.
- The prompt contains a few-shot example.
- LangGraph contains classify_intent.
- LangGraph contains retrieve_and_answer.
- LangGraph contains direct_answer.
- Conditional routing works for policy and general questions.
- Policy questions use retrieval.
- General questions use the direct-answer path.
- Pydantic validates answer, sources, and confidence.
- Mock mode works without an API key.
- FastAPI /ask works locally.
- Both retrieval and non-retrieval API calls were tested.
- Docker image builds successfully.
- Docker container starts FastAPI successfully.
- Docker API retrieval response was tested successfully.
- Docker API general-question response was tested successfully.

The complete Support Assistant is contained inside the /support_assistant module.


