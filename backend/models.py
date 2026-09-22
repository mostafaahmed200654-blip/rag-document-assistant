"""
Pydantic schemas shared by the FastAPI routes.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """A single retrieved chunk used to ground an answer."""

    document_id: str
    filename: str
    chunk_index: int
    text: str
    similarity_score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural language question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override number of chunks to retrieve")
    chat_history: Optional[List[dict]] = Field(
        default=None,
        description="Optional list of {'role': 'user'|'assistant', 'content': str} for follow-up questions",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    model_used: str
    chunks_retrieved: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    file_type: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    document_id: str
    status: str
    chunks_removed: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    ollama_model: str
    embedding_model: str
    vector_store_documents: int
    vector_store_chunks: int
