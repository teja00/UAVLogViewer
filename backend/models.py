from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class ChatRequest(BaseModel):
    message: str
    session_id: str
    file_data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime = datetime.now()

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    file_id: str
    metadata: Optional[Dict[str, Any]] = None

class TelemetryData(BaseModel):
    messages: Dict[str, Any]
    metadata: Dict[str, Any]
    start_time: Optional[datetime] = None
    vehicle_type: Optional[str] = None

class ConversationSession(BaseModel):
    session_id: str
    messages: List[ChatMessage] = []
    telemetry_data: Optional[TelemetryData] = None
    created_at: datetime = datetime.now()
    last_activity: datetime = datetime.now() 