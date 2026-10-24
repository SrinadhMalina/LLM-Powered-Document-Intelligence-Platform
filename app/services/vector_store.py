import uuid
import os
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


class VectorStoreService:
    """
    Production vector store backed by Qdrant.
    - Local mode  (default): persists to disk at QDRANT_PATH, no account needed.
    - Cloud mode  (optional): set QDRANT_URL + QDRANT_API_KEY env vars to use
                              Qdrant Cloud free tier (1 GB, always on).
    """

    COLLECTION_NAME = "doc_intel_collection"

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )

        # Switch between local and cloud transparently via env vars
        if settings.QDRANT_URL and settings.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            os.makedirs(settings.QDRANT_PATH, exist_ok=True)
            self.client = QdrantClient(path=settings.QDRANT_PATH)

        self._ensure_collection()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _ensure_collection(self):
        """Create the collection if it doesn't already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )

    @staticmethod
    def _make_id(doc_id: str) -> str:
        """Convert a string doc ID to a deterministic UUID (Qdrant requires UUID or int)."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

    # ── Public API ──────────────────────────────────────────────────────────

    def upsert_vectors(self, vectors: List[dict]):
        """
        Embed and store document chunks in Qdrant.
        Expected input: list of {'id': str, 'metadata': {'text': str, 'source': str, 'page': int}}
        """
        if not vectors:
            return

        texts = [v["metadata"]["text"] for v in vectors]
        embeddings = self.embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=self._make_id(v["id"]),
                vector=emb,
                payload=v["metadata"],
            )
            for v, emb in zip(vectors, embeddings)
        ]

        result = self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
            wait=True,
        )
        return result

    def query_vectors(self, query_text: str, top_k: int = 5) -> List[dict]:
        """
        Semantic search — returns top_k most relevant chunks.
        Scores are cosine similarity in [0, 1]; higher = more relevant.
        Uses query_points() (qdrant-client >= 1.7).
        """
        query_embedding = self.embeddings.embed_query(query_text)

        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "score": point.score,           # Cosine similarity [0, 1]
                "metadata": point.payload,
                "text": point.payload.get("text", ""),
            }
            for point in response.points
        ]

    def delete_document(self, source_filename: str):
        """Remove all vectors belonging to a specific source file."""
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
            ),
        )

    def get_collection_info(self) -> dict:
        """Return collection stats (useful for the API debug endpoint)."""
        count = self.client.count(
            collection_name=self.COLLECTION_NAME,
            exact=True,
        ).count
        info = self.client.get_collection(self.COLLECTION_NAME)
        return {
            "points_count": count,
            "status": str(info.status),
        }
