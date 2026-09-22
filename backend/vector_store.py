"""
Vector store layer: wraps a persistent ChromaDB collection and uses
Ollama (via its local embeddings endpoint) to embed text.
"""

from typing import Dict, List, Optional

import chromadb
import ollama
from chromadb.api.types import Documents, Embeddings


class OllamaEmbeddingFunction:
    """
    A Chroma-compatible embedding function that calls a local Ollama
    server's embedding endpoint (e.g. the 'nomic-embed-text' model).
    """

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's expected signature
        embeddings: Embeddings = []
        for text in input:
            response = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    def name(self) -> str:
        return f"ollama-{self.model}"


class VectorStore:
    """Thin, purpose-built interface over a single Chroma collection."""

    def __init__(self, persist_dir: str, collection_name: str, embed_model: str, ollama_base_url: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_function = OllamaEmbeddingFunction(model=embed_model, base_url=ollama_base_url)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        chunks: List[str],
    ) -> int:
        """Embed and store all chunks for a single document."""
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 4) -> List[Dict]:
        """Return the top_k most similar chunks to the query."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, count),
        )

        formatted = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # Chroma returns cosine *distance*; convert to a 0-1 similarity score.
            similarity = max(0.0, 1.0 - distance)
            formatted.append(
                {
                    "text": doc,
                    "document_id": meta.get("document_id"),
                    "filename": meta.get("filename"),
                    "chunk_index": meta.get("chunk_index"),
                    "similarity_score": round(similarity, 4),
                }
            )
        return formatted

    def list_documents(self) -> List[Dict]:
        """Aggregate stored chunks into a per-document summary."""
        all_items = self.collection.get(include=["metadatas"])
        metadatas = all_items.get("metadatas", [])

        docs: Dict[str, Dict] = {}
        for meta in metadatas:
            doc_id = meta.get("document_id")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename"),
                    "file_type": meta.get("file_type"),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns chunks removed."""
        existing = self.collection.get(where={"document_id": document_id}, include=[])
        ids_to_remove = existing.get("ids", [])
        if ids_to_remove:
            self.collection.delete(ids=ids_to_remove)
        return len(ids_to_remove)

    def total_chunks(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Danger: wipes the entire collection."""
        self.client.delete_collection(self.collection.name)
