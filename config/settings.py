import os
from pydantic_settings import BaseSettings # type: ignore

class Settings(BaseSettings):
    """Application settings."""
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "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")
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL: str = "llama-3.3-70b-specdec"  # Changed to match your current model
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()