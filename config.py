"""
Central configuration for the RAG Document Assistant.

All configuration is loaded from environment variables (or a local .env
file) using pydantic-settings. Import `settings` anywhere in the backend
to access strongly-typed configuration values.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # --- Chunking ---
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # --- Retrieval ---
    TOP_K_RESULTS: int = 4

    # --- Backend server ---
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # --- Uploads ---
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 25

    # --- Frontend ---
    BACKEND_API_URL: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
