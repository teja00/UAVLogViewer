"""
Configuration management for UAV Log Analyzer Backend
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Server Configuration  
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # CORS settings
    cors_origins: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080,http://localhost:8001"
    ).split(",")
    
    # Performance Configuration
    use_optimized_parser: bool = os.getenv("USE_OPTIMIZED_PARSER", "true").lower() == "true"
    parser_max_workers: int = int(os.getenv("PARSER_MAX_WORKERS", "0"))  # 0 means auto-detect

def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings() 