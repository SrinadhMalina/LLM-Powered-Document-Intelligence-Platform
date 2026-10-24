import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings


class IngestionService:
    """
    Loads PDFs, splits them into overlapping chunks, and prepares
    them for storage in the vector database.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,       # Smaller chunks = more precise retrieval
            chunk_overlap=150,    # Overlap preserves context across boundaries
            length_function=len,
            add_start_index=True,
        )

    def process_document(self, file_path: str) -> List[dict]:
        """
        Loads a PDF and returns a list of chunk dicts ready for VectorStoreService.

        Returns:
            List of {'id': str, 'metadata': {'text', 'source', 'page'}}
        """
        loader = PyPDFLoader(file_path)
        docs   = loader.load()
        chunks = self.text_splitter.split_documents(docs)

        processed = []
        source_name = os.path.basename(file_path)

        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if not text:          # skip empty chunks
                continue

            processed.append({
                "id": f"doc_{source_name}_{i}",
                "metadata": {
                    "text":   text,
                    "source": source_name,
                    "page":   chunk.metadata.get("page", 0),
                },
            })

        return processed
