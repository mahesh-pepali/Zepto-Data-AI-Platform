from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "zepto_policy"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(name=COLLECTION_NAME)

    query = "How can I track my order?"

    print(f"Query: {query}")
    print("Searching...")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    print()
    print("Top retrieved documents:")
    print("-" * 50)

    for i, document_id in enumerate(results["ids"][0]):
        print(f"Rank: {i + 1}")
        print(f"Document ID: {document_id}")
        print(f"Source: {results['metadatas'][0][i]['source']}")
        print(f"Distance: {results['distances'][0][i]}")
        print(f"Text: {results['documents'][0][i]}")
        print("-" * 50)


if __name__ == "__main__":
    main()