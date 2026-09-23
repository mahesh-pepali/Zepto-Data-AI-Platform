# Zepto Support Assistant



A policy-focused customer support assistant built with FastAPI, LangGraph, ChromaDB, and Sentence Transformers.



The assistant uses an offline mock LLM mode by default, so the complete baseline can be run without an API key.



## 1. Module Overview



The support assistant answers questions using a fixed Zepto policy corpus.



The module contains:



- 8 Zepto policy documents

- Sentence Transformer embeddings using `all-MiniLM-L6-v2`

- ChromaDB vector storage

- Cosine-similarity retrieval

- LangGraph workflow

- Structured prompt template

- Pydantic response validation

- FastAPI `/ask` endpoint

- Docker support

- Optional real LLM mode



The default mode is:



```text

MOCK\_LLM=1

