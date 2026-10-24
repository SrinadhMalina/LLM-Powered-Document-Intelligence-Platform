from fastapi import APIRouter, UploadFile, File, HTTPException
from functools import lru_cache
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStoreService
from app.services.query_engine import QueryEngine
from app.core.cache import ResponseCache
import shutil
import os

router = APIRouter()


# ── Lazy singletons ─────────────────────────────────────────────────────────
# Services are created on FIRST REQUEST, not at import time.
# This prevents Qdrant file-lock conflicts when starting multiple processes.

@lru_cache(maxsize=1)
def _ingestion() -> IngestionService:
    return IngestionService()

@lru_cache(maxsize=1)
def _vector_store() -> VectorStoreService:
    return VectorStoreService()

@lru_cache(maxsize=1)
def _query_engine() -> QueryEngine:
    return QueryEngine(vector_store=_vector_store())  # share one Qdrant connection

@lru_cache(maxsize=1)
def _cache() -> ResponseCache:
    return ResponseCache()


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Check API and vector DB status."""
    try:
        info = _vector_store().get_collection_info()
        return {"status": "online", "vector_store": info}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, chunk it, embed it, and store in Qdrant."""
    os.makedirs("data/uploads", exist_ok=True)
    file_path = f"data/uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        vectors = _ingestion().process_document(file_path)
        _vector_store().upsert_vectors(vectors)
        return {
            "message": f"Document '{file.filename}' indexed successfully.",
            "chunks":  len(vectors),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/document/{filename}")
async def delete_document(filename: str):
    """Remove all vectors for a given source file."""
    try:
        _vector_store().delete_document(filename)
        return {"message": f"Deleted all vectors for '{filename}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_docs(question: str):
    """Ask a question against the indexed documents (RAG pipeline)."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        cached = _cache().get(question)
        if cached:
            return {**cached, "cached": True}

        result = _query_engine().answer_question(question)
        _cache().set(question, result)
        return {**result, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
