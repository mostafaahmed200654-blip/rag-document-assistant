"""
FastAPI backend for the RAG Document Assistant.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.document_processor import SUPPORTED_EXTENSIONS, get_file_extension
from backend.llm_service import LLMService
from backend.models import (
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from backend.rag_engine import RAGEngine
from backend.vector_store import VectorStore
from config import settings

app = FastAPI(
    title="RAG Document Assistant API",
    description="Retrieval-Augmented Generation API for querying uploaded documents using local LLMs via Ollama.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

vector_store = VectorStore(
    persist_dir=settings.CHROMA_PERSIST_DIR,
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embed_model=settings.OLLAMA_EMBED_MODEL,
    ollama_base_url=settings.OLLAMA_BASE_URL,
)

llm_service = LLMService(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)

rag_engine = RAGEngine(
    vector_store=vector_store,
    llm_service=llm_service,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    top_k=settings.TOP_K_RESULTS,
)


@app.get("/", tags=["General"])
def root():
    return {
        "service": "RAG Document Assistant API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    doc_summary = rag_engine.list_documents()
    return HealthResponse(
        status="ok",
        ollama_reachable=llm_service.is_reachable(),
        ollama_model=settings.OLLAMA_MODEL,
        embedding_model=settings.OLLAMA_EMBED_MODEL,
        vector_store_documents=doc_summary["total_documents"],
        vector_store_chunks=doc_summary["total_chunks"],
    )


@app.post("/upload", response_model=IngestResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    extension = get_file_extension(file.filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    temp_filename = f"{uuid.uuid4()}{extension}"
    temp_path = os.path.join(settings.UPLOAD_DIR, temp_filename)

    size = 0
    try:
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds max upload size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                    )
                buffer.write(chunk)

        result = rag_engine.ingest_document(file_path=temp_path, filename=file.filename)
        return IngestResponse(**result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_documents(request: QueryRequest):
    try:
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            chat_history=request.chat_history,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer query: {e}")


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    result = rag_engine.list_documents()
    return DocumentListResponse(**result)


@app.delete("/documents/{document_id}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(document_id: str):
    result = rag_engine.delete_document(document_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"No document found with id '{document_id}'.")
    return DeleteResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, reload=True)
