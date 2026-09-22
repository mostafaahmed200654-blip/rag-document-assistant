"""
RAG engine: the top-level orchestrator that ties document processing,
the vector store, and the LLM service together.
"""

import os
import uuid
from typing import Dict, List, Optional

from backend.document_processor import get_file_extension, process_document
from backend.llm_service import LLMService
from backend.vector_store import VectorStore


class RAGEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: LLMService,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def ingest_document(self, file_path: str, filename: str) -> Dict:
        """Process a saved file on disk and index it into the vector store."""
        file_type = get_file_extension(filename)
        chunks = process_document(
            file_path=file_path,
            filename=filename,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        document_id = str(uuid.uuid4())
        chunks_created = self.vector_store.add_chunks(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            chunks=chunks,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": chunks_created,
            "status": "indexed",
        }

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        chat_history: Optional[List[dict]] = None,
    ) -> Dict:
        """Retrieve relevant chunks and generate a grounded answer."""
        k = top_k or self.top_k
        retrieved_chunks = self.vector_store.query(query_text=question, top_k=k)

        answer = self.llm_service.generate_answer(
            question=question,
            context_chunks=retrieved_chunks,
            chat_history=chat_history,
        )

        sources = [
            {
                "document_id": c["document_id"],
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "similarity_score": c["similarity_score"],
            }
            for c in retrieved_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "model_used": self.llm_service.model,
            "chunks_retrieved": len(retrieved_chunks),
        }

    def list_documents(self) -> Dict:
        documents = self.vector_store.list_documents()
        total_chunks = sum(d["chunk_count"] for d in documents)
        return {
            "documents": documents,
            "total_documents": len(documents),
            "total_chunks": total_chunks,
        }

    def delete_document(self, document_id: str) -> Dict:
        removed = self.vector_store.delete_document(document_id)
        return {
            "document_id": document_id,
            "status": "deleted" if removed > 0 else "not_found",
            "chunks_removed": removed,
        }
