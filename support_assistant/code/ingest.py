import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "zepto_policy"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_policy_documents():
    documents = []
    ids = []
    metadatas = []

    for file_path in sorted(CORPUS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(text)
        ids.append(file_path.stem)
        metadatas.append(
            {
                "document_id": file_path.stem,
                "source": file_path.name,
            }
        )

    return documents, ids, metadatas


def build_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading policy documents...")
    documents, ids, metadatas = load_policy_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 policy documents, but found {len(documents)}."
        )

    print(f"Found {len(documents)} policy documents.")

    print("Creating ChromaDB client...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Zepto policy support corpus"},
    )

    print("Creating embeddings...")
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print()
    print("ChromaDB ingestion completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents stored: {collection.count()}")
    print(f"Database path: {CHROMA_DIR}")


if __name__ == "__main__":
    build_collection()