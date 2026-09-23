from fastapi import FastAPI
from pydantic import BaseModel, Field

from support_assistant.code.graph import run_support_graph


app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline policy support assistant using LangGraph and ChromaDB.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running."
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = run_support_graph(request.query)

    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),)