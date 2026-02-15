from __future__ import annotations

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "LocalRAG"

    chroma_dir: str = "backend/.chroma"
    chroma_collection: str = "documents"

    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k_default: int = 4

    llm_provider: str = "ollama"  # ollama | llamacpp
    llm_model: str = Field(default="llama3.1:latest", validation_alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_URL"
    )
    ollama_timeout_s: float = Field(default=30.0, validation_alias="OLLAMA_TIMEOUT")
    ollama_retries: int = Field(default=5, validation_alias="OLLAMA_RETRIES")
    llamacpp_base_url: str = "http://localhost:8080"

    embeddings_provider: str = "ollama"  # ollama | st
    embeddings_model: str = "nomic-embed-text"
    embeddings_st_model: str = "all-MiniLM-L6-v2"

    log_level: str = "INFO"


settings = Settings()
