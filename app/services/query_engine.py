from langchain_groq import ChatGroq
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from typing import List, Dict, Optional


class QueryEngine:
    """
    RAG query engine:
      1. Retrieves relevant chunks from Qdrant via semantic search.
      2. Feeds them as context to the Groq LLM.
      3. Returns the answer with source attribution.

    Automatic model fallback: cycles through GROQ_MODEL_FALLBACKS if the
    primary model is unavailable (handles decommissioned or restricted models).
    """

    # Minimum relevance score — chunks below this are too noisy to use
    RELEVANCE_THRESHOLD = 0.30

    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        # Accept a shared instance to avoid opening a second Qdrant connection
        self.vector_store = vector_store if vector_store is not None else VectorStoreService()
        self._model_index = 0
        self._init_llm()

    # ── LLM setup ───────────────────────────────────────────────────────────

    def _init_llm(self):
        all_models = settings.GROQ_MODEL_FALLBACKS
        model_name = all_models[self._model_index % len(all_models)]
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=model_name,
            temperature=0,
            max_tokens=1024,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def answer_question(self, question: str) -> Dict:
        """
        Run the full RAG pipeline and return {'answer': str, 'sources': list}.
        """
        # 1. Retrieve context
        matches = self.vector_store.query_vectors(question, top_k=5)

        # Filter low-relevance chunks
        matches = [m for m in matches if m["score"] >= self.RELEVANCE_THRESHOLD]

        context_text = ""
        sources      = []

        for match in matches:
            meta = match["metadata"]
            context_text += (
                f"Source: {meta['source']} (Page {int(meta['page']) + 1})\n"
                f"Content: {match['text']}\n\n"
            )
            sources.append({
                "source": meta["source"],
                "page":   int(meta["page"]),
                "score":  round(match["score"], 4),
            })

        # 2. Build prompt
        if context_text:
            prompt = f"""You are an expert document intelligence assistant.
Answer the question using ONLY the context provided below.
Always cite the source filename and page number in your answer.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context_text}
Question: {question}

Answer:"""
        else:
            prompt = (
                f"No relevant documents were found for the question: '{question}'.\n"
                "Please upload and index relevant documents first."
            )
            return {"answer": prompt, "sources": []}

        # 3. Call Groq with automatic model fallback
        all_models = settings.GROQ_MODEL_FALLBACKS
        while self._model_index < len(all_models):
            try:
                response = self.llm.invoke(prompt)
                return {"answer": response.content, "sources": sources}
            except Exception as e:
                print(f"[QueryEngine] Model '{self.llm.model_name}' failed: {e}")
                self._model_index += 1
                if self._model_index < len(all_models):
                    print(f"[QueryEngine] Falling back to '{all_models[self._model_index]}'…")
                    self._init_llm()

        # All models exhausted
        return {
            "answer": "All AI models are currently unavailable. Please try again later.",
            "sources": sources,
        }
