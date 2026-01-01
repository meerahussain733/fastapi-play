import threading
import logging
import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from model import ask_llm, MODEL_ID

from rag.loader import load_txt_documents
from rag.store import RAGStore
from rag.retriever import retrieve_top_k
from rag.embedder import embed_texts
from rag.prompting import build_rag_prompt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create store (build happens in lifespan startup)
store = RAGStore(docs_dir="docs", chunk_size=800, overlap=100)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when server starts
    store.build()
    yield
    # Optional: cleanup on shutdown (nothing needed for now)

app = FastAPI(title="FastAPI Local LLM Backend", version="0.1.0", lifespan=lifespan)

lock = threading.Lock()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    try:
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        logger.info(
            f"id={request_id} {request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)"
        )
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        logger.exception(
            f"id={request_id} {request.method} {request.url.path} -> 500 ({duration_ms}ms) error={e}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/ask")
def ask(req: AskRequest):
    """
    Week 2 endpoint: LLM only (no retrieval)
    """
    try:
        logger.info(f"/ask question: {req.question[:80]!r}")

        start = time.time()
        with lock:
            answer = ask_llm(req.question)
        duration_ms = int((time.time() - start) * 1000)

        logger.info(f"LLM generation took {duration_ms}ms")
        return {"answer": answer, "model": MODEL_ID}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="LLM inference failed")


@app.get("/knowledge")
def list_docs():
    docs = load_txt_documents("docs")
    return {"count": len(docs), "sources": [d["source"] for d in docs]}


@app.get("/search")
def search(q: str, k: int = 3):
    results = retrieve_top_k(q, store.chunks, store.vectors, embed_texts, k=k)
    return {
        "query": q,
        "k": k,
        "matches": [
            {
                "source": r["source"],
                "chunk_id": r["chunk_id"],
                "score": round(r["score"], 4),
                "preview": r["text"][:200],
            }
            for r in results
        ],
    }


@app.get("/ask-rag")
def ask_rag(q: str, k: int = 3, min_score: float = 0.25):
    """
    Week 3 endpoint: Retrieval + LLM (grounded).
    If top score < min_score -> returns "I don't know..." without using context.
    """
    matches = retrieve_top_k(q, store.chunks, store.vectors, embed_texts, k=k)

    if not matches or matches[0]["score"] < min_score:
        return {
            "question": q,
            "k": k,
            "min_score": min_score,
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "top_score": round(matches[0]["score"], 4) if matches else None,
            "used_context": False,
        }

    prompt = build_rag_prompt(q, matches)

    with lock:
        answer = ask_llm(prompt)

    return {
        "question": q,
        "k": k,
        "min_score": min_score,
        "answer": answer,
        "sources": [
            {"source": m["source"], "chunk_id": m["chunk_id"], "score": round(m["score"], 4)}
            for m in matches
        ],
        "top_score": round(matches[0]["score"], 4),
        "used_context": True,
    }


@app.post("/reindex")
def reindex():
    """
    Rebuild docs -> chunks -> embeddings without restarting the server.
    """
    store.build()
    return {"status": "ok", "chunks": len(store.chunks)}
