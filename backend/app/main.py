from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag import ask, get_vectorstore
from .settings import settings

app = FastAPI(title="Clinical QA RAG (FastAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskPayload(BaseModel):
    question: str
    session_id: str = "default"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/index/rebuild")
def rebuild_index():
    # Rebuild/persist Chroma index (optional trigger)
    vs = get_vectorstore()
    return {"ok": True, "doc_count": vs._collection.count()}

@app.post("/ask")
def ask_route(payload: AskPayload):
    answer, citations = ask(payload.question, payload.session_id)
    return {
        "answer": answer,
        "citations": citations,
        "followups": [],      # You can add auto-suggestions here
        "confidence": 0.7     # Heuristic
    }