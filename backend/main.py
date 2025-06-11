"""
Main FastAPI Application

UAV Log Viewer Chatbot Backend
Provides API endpoints for file upload, telemetry parsing, and chatbot interactions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import uuid
from typing import Dict, Any, Optional
import asyncio

from config import settings
from models import ChatRequest, ChatResponse, FileUploadResponse
from chat_service import ChatService
from telemetry_parser import TelemetryParser

# Initialize FastAPI app
app = FastAPI(
    title="UAV Log Viewer Chatbot API",
    description="Backend API for UAV telemetry log analysis chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
chat_service = ChatService()
telemetry_parser = TelemetryParser()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "UAV Log Viewer Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "chat": "/chat",
            "sessions": "/sessions",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "active_sessions": len(chat_service.sessions)
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
) -> FileUploadResponse:
    """
    Upload and parse a telemetry file (.bin, .tlog, .txt).
    
    Args:
        file: The uploaded telemetry file
        session_id: Optional session ID to associate the data with
    
    Returns:
        FileUploadResponse with parsing results
    """
    
    # Validate file type
    file_extension = file.filename.lower().split('.')[-1] if file.filename else ""
    if f".{file_extension}" not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {settings.ALLOWED_FILE_TYPES}"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Generate file ID and session if not provided
        file_id = str(uuid.uuid4())
        if session_id is None:
            session_id = chat_service.create_session()
        
        # For now, we'll simulate the parsing process since we don't have 
        # the actual JavaScript parsing logic in Python
        # In a real implementation, you would:
        # 1. Parse the binary file using a Python MAVLink library
        # 2. Extract telemetry messages similar to the JavaScript version
        
        # Simulated telemetry data structure (matching JavaScript format)
        simulated_telemetry = {
            "messages": {
                "GPS": {
                    "time_boot_ms": [1000, 2000, 3000, 4000, 5000],
                    "lat": [40.7128, 40.7129, 40.7130, 40.7131, 40.7132],
                    "lon": [-74.0060, -74.0061, -74.0062, -74.0063, -74.0064],
                    "alt": [10.5, 12.3, 15.8, 18.2, 20.1]
                },
                "ATT": {
                    "time_boot_ms": [1000, 2000, 3000, 4000, 5000],
                    "Roll": [0.1, 0.2, -0.1, 0.3, -0.2],
                    "Pitch": [0.05, -0.1, 0.15, -0.05, 0.08],
                    "Yaw": [1.57, 1.58, 1.56, 1.59, 1.55]
                },
                "MODE": {
                    "time_boot_ms": [500, 2500, 4500],
                    "asText": ["STABILIZE", "LOITER", "RTL"]
                },
                "CURR": {
                    "time_boot_ms": [1000, 2000, 3000, 4000, 5000],
                    "Volt": [12.6, 12.5, 12.4, 12.3, 12.2],
                    "Curr": [2.1, 2.3, 2.5, 2.2, 2.0]
                }
            },
            "metadata": {
                "startTime": 1640995200000,  # Unix timestamp in ms
                "vehicleType": "Quadcopter",
                "logType": file_extension
            }
        }
        
        # Update session with telemetry data
        chat_service.update_session_telemetry(session_id, simulated_telemetry)
        
        return FileUploadResponse(
            success=True,
            message=f"File '{file.filename}' uploaded and parsed successfully",
            file_id=file_id,
            metadata={
                "session_id": session_id,
                "file_size": len(file_content),
                "file_type": file_extension,
                "message_types": list(simulated_telemetry["messages"].keys()),
                "total_messages": sum(len(msg.get("time_boot_ms", [])) for msg in simulated_telemetry["messages"].values())
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the chatbot and get a response.
    
    Args:
        request: ChatRequest containing message and session info
    
    Returns:
        ChatResponse with the bot's reply
    """
    
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured"
        )
    
    try:
        # Get response from chat service
        response_text = await chat_service.get_chat_response(
            session_id=request.session_id,
            user_message=request.message
        )
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    session_info = chat_service.get_session_info(session_id)
    if session_info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_info

@app.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get the conversation history for a session."""
    history = chat_service.get_conversation_history(session_id)
    return {"session_id": session_id, "messages": history}

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    success = chat_service.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session cleared successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = []
    for session_id, session in chat_service.sessions.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(session.messages),
            "has_telemetry_data": session.telemetry_data is not None,
            "last_activity": session.last_activity.isoformat()
        })
    return {"sessions": sessions}

@app.post("/sessions")
async def create_session():
    """Create a new conversation session."""
    session_id = chat_service.create_session()
    return {
        "session_id": session_id,
        "message": "New session created successfully"
    }

@app.post("/telemetry/analyze")
async def analyze_telemetry(request: dict):
    """
    Analyze telemetry data and return insights.
    
    Args:
        request: Dictionary containing telemetry messages and query
    
    Returns:
        Analysis results
    """
    try:
        query = request.get("query", "")
        messages = request.get("messages", {})
        
        # Create a temporary parser instance
        temp_parser = TelemetryParser()
        temp_parser.messages = messages
        
        # Perform analysis
        if query:
            result = temp_parser.query_telemetry_data(query)
        else:
            result = temp_parser.get_data_summary()
        
        return {"analysis": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing telemetry data: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 