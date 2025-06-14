"""
FastAPI backend for UAV Log Analyzer Chatbot V2

This backend uses the enhanced agentic V2 chat service for analyzing UAV flight telemetry data.
It processes uploaded log files directly and provides AI-powered analysis using OpenAI GPT models.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uuid
import logging
import tempfile
import os
import sys
import uvicorn
from dotenv import load_dotenv

from config import get_settings
from models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ErrorResponse
)
# Import the new multi-role agent
from agent.multi_role_agent import MultiRoleAgent

# Create a wrapper to maintain compatibility with existing API
class ChatServiceV2:
    """Compatibility wrapper for the multi-role agent."""
    
    def __init__(self):
        self.agent = MultiRoleAgent()
        
    async def create_or_get_session(self, session_id=None):
        return await self.agent.create_or_get_session(session_id)
        
    async def process_log_file(self, session_id: str, file_path: str):
        await self.agent.process_log_file(session_id, file_path)
        
    async def chat(self, session_id: str, user_message: str) -> str:
        return await self.agent.chat(session_id, user_message)
        
    @property
    def sessions(self):
        return self.agent.sessions
    
    def get_performance_stats(self, session_id=None):
        """Get performance statistics from the multi-role agent."""
        return self.agent.get_performance_stats(session_id)

# Initialize the multi-role chat service
chat_service_v2 = ChatServiceV2()

# Load environment variables early
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UAV Log Analyzer Chat API V2",
    description="Multi-role AI-powered chatbot with Planner→Executor→Critic architecture for analyzing UAV flight telemetry data",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get settings
settings = get_settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Log and return detailed validation errors from Pydantic.
    """
    error_details = exc.errors()
    logger.error(f"Validation error for request to {request.url}: {error_details}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed", "errors": error_details}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        openai_configured=bool(settings.openai_api_key)
    )


# === UPLOAD LOG ENDPOINTS ===

@app.post("/sessions/upload-log", summary="Upload a log file for analysis")
async def upload_log_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Uploads a flight log file (.bin, .tlog), starts a background processing
    task to convert it into pandas DataFrames, and returns a session ID
    for the chat service.
    """
    session_id = str(uuid.uuid4())
    
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{file.filename}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        logger.info(f"[{session_id}] Received log file '{file.filename}', saved to '{tmp_path}'")

        # Create the session and immediately mark it as processing
        session = await chat_service_v2.create_or_get_session(session_id)
        session.is_processing = True
        
        # Add background task to process the file and then delete it
        background_tasks.add_task(chat_service_v2.process_log_file, session_id, tmp_path)
        background_tasks.add_task(os.unlink, tmp_path)
        
        return {"session_id": session_id, "message": "File upload successful, processing has started."}

    except Exception as e:
        logger.error(f"File upload failed: {str(e)}", exc_info=True)
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {str(e)}"
        )


# V2 alias for backward compatibility
@app.post("/v2/sessions/upload-log", summary="Upload a log file for analysis (V2 alias)")
async def upload_log_file_v2_alias(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """V2 alias for backward compatibility"""
    return await upload_log_file(background_tasks, file)


# === CHAT ENDPOINTS ===

@app.post("/chat", response_model=ChatResponse, summary="Chat with V2 Agentic Service")
async def chat(
    chat_request: ChatRequest,
):
    """
    Agentic chat endpoint that uses tool-calling to analyze telemetry data.
    Assumes that telemetry data has already been loaded into the session
    using the `/sessions/upload-log` endpoint.
    """
    try:
        if not chat_request.session_id:
            raise HTTPException(status_code=400, detail="A 'session_id' is required for chat.")

        # Get AI response from the V2 service
        response_text = await chat_service_v2.chat(chat_request.session_id, chat_request.message)
        
        # Get session info
        session = await chat_service_v2.create_or_get_session(chat_request.session_id)
        conversation_count = len(session.messages)

        return ChatResponse(
            response=response_text,
            session_id=chat_request.session_id,
            conversation_count=conversation_count
        )
        
    except Exception as e:
        logger.error("Chat endpoint error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


# V2 alias for backward compatibility
@app.post("/v2/chat", response_model=ChatResponse, summary="Chat with V2 Agentic Service (V2 alias)")
async def chat_v2_alias(
    chat_request: ChatRequest,
):
    """V2 alias for backward compatibility"""
    return await chat(chat_request)


# === SESSION MANAGEMENT ENDPOINTS ===

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """
    Get information about a chat session including processing status and dataframes.
    """
    if session_id not in chat_service_v2.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_service_v2.sessions[session_id]
    
    return {
        "session_id": session.session_id,
        "message_count": len(session.messages),
        "has_dataframes": bool(session.dataframes),
        "dataframe_count": len(session.dataframes),
        "dataframe_types": list(session.dataframes.keys()) if session.dataframes else [],
        "is_processing": session.is_processing,
        "processing_error": session.processing_error,
        "created_at": session.created_at.isoformat(),
        "last_updated": session.last_updated.isoformat()
    }


# V2 alias for backward compatibility
@app.get("/v2/sessions/{session_id}")
async def get_session_info_v2_alias(session_id: str):
    """V2 alias for backward compatibility"""
    return await get_session_info(session_id)


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    Clear conversation history for a session (keeps telemetry data).
    """
    if session_id not in chat_service_v2.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = chat_service_v2.sessions[session_id]
        # Clear messages but keep dataframes and other session data
        session.messages = []
        session.last_updated = datetime.now()
        
        return {"message": "Session history cleared successfully"}
        
    except Exception as e:
        logger.error(f"Session clear error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear session: {str(e)}"
        )


# V2 alias for backward compatibility
@app.delete("/v2/sessions/{session_id}")
async def clear_session_v2_alias(session_id: str):
    """V2 alias for backward compatibility"""
    return await clear_session(session_id)


# === PERFORMANCE MONITORING ENDPOINTS ===

@app.get("/performance/stats")
async def get_overall_performance_stats():
    """Get overall performance statistics for the multi-role agent."""
    try:
        stats = chat_service_v2.get_performance_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance stats: {str(e)}"
        )


@app.get("/performance/stats/{session_id}")
async def get_session_performance_stats(session_id: str):
    """Get performance statistics for a specific session."""
    try:
        if session_id not in chat_service_v2.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        stats = chat_service_v2.get_performance_stats(session_id)
        return {
            "status": "success",
            "session_id": session_id,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session performance stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session performance stats: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("Unhandled exception", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc)
        ).dict()
    )


def start_server():
    """Start the FastAPI server with proper configuration."""
    
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("Please set your OpenAI API key in a .env file or environment variables.")
        print("Copy env.example to .env and add your API key.")
        print()
        
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print("Starting UAV Log Viewer Chatbot Backend V2...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"ReDoc Documentation: http://{host}:{port}/redoc")
    print("")
    
    try:
        app_import = "main:app"
        if reload:
            uvicorn.run(app_import, host=host, port=port, reload=True, log_level=log_level)
        else:
            uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_server() 