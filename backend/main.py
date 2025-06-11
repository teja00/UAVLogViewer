"""
FastAPI backend for UAV Log Analyzer Chatbot

This backend receives parsed telemetry data from the frontend and provides
AI-powered analysis using OpenAI GPT models with ArduPilot knowledge.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import uuid
import logging
import tempfile
import os

from config import get_settings
from models import (
    ChatRequest,
    ChatResponse,
    TelemetryData,
    HealthResponse,
    ErrorResponse,
    ConversationSession
)
from chat_service import chat_service
from chat_service_v2 import chat_service_v2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UAV Log Analyzer Chat API",
    description="AI-powered chatbot for analyzing UAV flight telemetry data",
    version="2.0.0",
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


# Background task to cleanup old sessions
async def cleanup_sessions():
    """Background task to cleanup old chat sessions"""
    chat_service.cleanup_old_sessions(hours=24)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        openai_configured=bool(settings.openai_api_key)
    )


# --- V1 Endpoints ---

@app.post("/chat", response_model=ChatResponse, summary="Chat with Standard V1 Service")
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Main chat endpoint that receives user messages and telemetry data from frontend.
    This uses the standard, non-agentic chat service.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Clear history if requested
        if request.clear_history:
            await chat_service.clear_session_history(session_id)
            
        # Update session with telemetry data if provided
        if request.telemetry_data:
            await chat_service.update_session_telemetry(session_id, request.telemetry_data)
            
        # Get AI response
        response_text = await chat_service.chat(session_id, request.message)
        
        # Get session info
        session = chat_service.get_session(session_id)
        conversation_count = len(session.messages) if session else 0
        
        # Schedule cleanup task
        background_tasks.add_task(cleanup_sessions)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            conversation_count=conversation_count
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@app.post("/sessions/{session_id}/telemetry", summary="Upload Telemetry for V1 Service")
async def update_session_telemetry(
    session_id: str,
    telemetry_data: TelemetryData
):
    """
    Update telemetry data for an existing V1 session.
    """
    try:
        await chat_service.update_session_telemetry(session_id, telemetry_data)
        return {"message": "Telemetry data updated successfully"}
        
    except Exception as e:
        logger.error(f"Telemetry update error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update telemetry data: {str(e)}"
        )


# --- V2 Agentic Chat Endpoints ---

@app.post("/v2/sessions/upload-log", summary="Upload a log file for V2 analysis")
async def upload_log_file_v2(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Uploads a flight log file (.bin, .tlog), starts a background processing
    task to convert it into pandas DataFrames, and returns a session ID
    for the V2 chat service.
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


@app.post("/v2/chat", response_model=ChatResponse, summary="Chat with V2 Agentic Service")
async def chat_v2(
    chat_request: ChatRequest,
):
    """
    Agentic chat endpoint that uses tool-calling to analyze telemetry data.
    Assumes that telemetry data has already been loaded into the session
    using the `/v2/sessions/upload-log` endpoint.
    The `telemetry_data` field in the request body is ignored.
    """
    try:
        if not chat_request.session_id:
            raise HTTPException(status_code=400, detail="A 'session_id' is required for V2 chat.")

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
        logger.error("V2 Chat endpoint error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process V2 chat request: {str(e)}"
        )


# --- Common Session Endpoints ---

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """
    Get information about a chat session. Note: This currently only inspects V1 sessions.
    """
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return {
        "session_id": session.session_id,
        "message_count": len(session.messages),
        "has_telemetry_data": session.telemetry_data is not None,
        "created_at": session.created_at.isoformat(),
        "last_updated": session.last_updated.isoformat()
    }


@app.get("/v2/sessions/{session_id}")
async def get_v2_session_info(session_id: str):
    """
    Get information about a V2 chat session including processing status and dataframes.
    """
    if session_id not in chat_service_v2.sessions:
        raise HTTPException(status_code=404, detail="V2 Session not found")
    
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


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    Clear conversation history for a V1 session (keeps telemetry data).
    """
    try:
        await chat_service.clear_session_history(session_id)
        return {"message": "Session history cleared successfully"}
        
    except Exception as e:
        logger.error(f"Session clear error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear session: {str(e)}"
        )


@app.post("/test/simulate-frontend-data")
async def simulate_frontend_data():
    """
    Test endpoint that simulates frontend-parsed telemetry data.
    This helps test the backend without needing the full frontend.
    """
    
    # Simulate typical frontend-parsed telemetry data structure
    simulated_data = TelemetryData(
        messages={
            "GPS": {
                "time_boot_ms": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                "lat": [40.7128, 40.7129, 40.7130, 40.7131, 40.7132, 40.7133, 40.7134, 40.7135, 40.7136, 40.7137],
                "lng": [-74.0060, -74.0061, -74.0062, -74.0063, -74.0064, -74.0065, -74.0066, -74.0067, -74.0068, -74.0069],
                "alt": [10.5, 12.3, 15.8, 18.2, 20.1, 22.5, 25.0, 27.3, 30.1, 32.8]
            },
            "ATT": {
                "time_boot_ms": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                "Roll": [0.1, 0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.1, 0.2, -0.1],
                "Pitch": [0.05, -0.1, 0.15, -0.05, 0.08, -0.12, 0.18, -0.08, 0.06, 0.02],
                "Yaw": [1.57, 1.58, 1.56, 1.59, 1.55, 1.60, 1.54, 1.58, 1.57, 1.56]
            }
        },
        metadata={
            "startTime": 1640995200000,
            "vehicleType": "Quadcopter",
            "logType": "bin",
            "duration": 10.0,
            "messageCount": 42
        }
    )
    
    return {
        "message": "Simulated telemetry data generated",
        "data": simulated_data.dict(),
    }


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 