import json
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

from support_assistant.code.prompt import build_prompt


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "zepto_policy"
MODEL_NAME = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = {
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
}


class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class SupportState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float
    retrieved_documents: list[str]
    retrieved_ids: list[str]
    prompt: str


print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(name=COLLECTION_NAME)


def validate_response(
    answer: str,
    sources: list[str],
    confidence: float,
) -> SupportResponse:
    return SupportResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
    )


def call_real_llm(
    prompt: str,
    expected_sources: list[str],
) -> SupportResponse:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when MOCK_LLM=0."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )

    current_prompt = (
        prompt
        + "\n\nReturn ONLY valid JSON with exactly these fields:\n"
        + '{"answer": "string", "confidence": 0.0}\n'
        + "Do not include Markdown or additional fields."
    )

    last_error = None

    for attempt in range(3):
        try:
            raw_response = llm.invoke(current_prompt)

            raw_content = raw_response.content

            if isinstance(raw_content, list):
                raw_content = "".join(
                    str(item) for item in raw_content
                )

            parsed = json.loads(raw_content)

            validated = SupportResponse(
                answer=str(parsed["answer"]),
                sources=expected_sources,
                confidence=float(parsed["confidence"]),
            )

            return validated

        except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as exc:
            last_error = exc

            if attempt < 2:
                current_prompt = (
                    prompt
                    + "\n\nYour previous response failed JSON schema validation."
                    + "\nReturn ONLY valid JSON with exactly these fields:"
                    + '\n{"answer": "string", "confidence": 0.0}'
                    + "\nThe confidence value must be between 0 and 1."
                    + "\nDo not include Markdown or additional fields."
                )

    raise RuntimeError(
        f"LLM response failed schema validation after 3 attempts: {last_error}"
    )


def classify_intent(state: SupportState) -> SupportState:
    query = state["query"].lower()

    is_policy_question = any(
        keyword in query for keyword in POLICY_KEYWORDS
    )

    if is_policy_question:
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent": intent,
    }


def retrieve_and_answer(state: SupportState) -> SupportState:
    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    retrieved_documents = results["documents"][0]
    retrieved_ids = results["ids"][0]

    top_chunk = retrieved_documents[0]
    top_chunk_snippet = top_chunk[:200]

    context = "\n\n".join(retrieved_documents)

    prompt = build_prompt(
        query=query,
        context=context,
    )

    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm != "0":
        response = validate_response(
            answer=f"Based on the retrieved context: {top_chunk_snippet}",
            sources=retrieved_ids,
            confidence=1.0,
        )
    else:
        try:
            response = call_real_llm(
                prompt=prompt,
                expected_sources=retrieved_ids,
            )
        except RuntimeError as exc:
            response = validate_response(
                answer=f"Unable to generate a validated answer: {exc}",
                sources=retrieved_ids,
                confidence=0.0,
            )

    return {
        **state,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "retrieved_documents": retrieved_documents,
        "retrieved_ids": retrieved_ids,
        "prompt": prompt,
    }


def direct_answer(state: SupportState) -> SupportState:
    mock_llm = os.getenv("MOCK_LLM", "1")

    if mock_llm != "0":
        response = validate_response(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        try:
            response = call_real_llm(
                prompt=(
                    "You are a Zepto customer support assistant.\n"
                    "Answer the customer's question directly.\n"
                    "Do not retrieve or invent Zepto policy information.\n"
                    f"Customer question: {state['query']}"
                ),
                expected_sources=[],
            )
        except RuntimeError as exc:
            response = validate_response(
                answer=f"Unable to generate a validated answer: {exc}",
                sources=[],
                confidence=0.0,
            )

    return {
        **state,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
    }


def route_after_classification(state: SupportState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def build_graph():
    workflow = StateGraph(SupportState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


support_graph = build_graph()


def run_support_graph(query: str) -> SupportState:
    return support_graph.invoke(
        {
            "query": query,
        }
    )