import threading
import logging
import uuid
from fastapi import Request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
from model import ask_llm, MODEL_ID
from fastapi.responses import JSONResponse
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

App = FastAPI(title="FastAPI Local LLM Backend", version="0.1.0")

lock = threading.Lock()

@App.middleware("http")
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


@App.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@App.post("/ask")
def ask(req: AskRequest):
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

