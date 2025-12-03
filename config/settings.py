import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # OpenAI
    openai_api_key: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_products_collection: str = "scicon_products"

    # Embeddings
    embedding_model: str = "text-embedding-3-large"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
