#!/usr/bin/env python3
"""
Startup script for UAV Log Viewer Chatbot Backend

This script starts the FastAPI server with proper configuration.
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

def main():
    """Start the FastAPI server."""
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("Please set your OpenAI API key in a .env file or environment variables.")
        print("Copy env.example to .env and add your API key.")
        
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"Starting UAV Log Viewer Chatbot Backend...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"ReDoc Documentation: http://{host}:{port}/redoc")
    print("")
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 