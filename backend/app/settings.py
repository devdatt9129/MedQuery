from pydantic import BaseModel
import os

# ✅ load .env at import time
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseModel):
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "openai")  # or "e5"
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    HF_E5_MODEL: str = os.getenv("HF_E5_MODEL", "intfloat/e5-base-v2")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*")

settings = Settings()