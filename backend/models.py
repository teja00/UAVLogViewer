"""
Pydantic models for UAV Log Analyzer Chat API V2
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class ChatRequest(BaseModel):
    """Request to chat endpoint"""
    message: str = Field(..., description="User's question about the telemetry data")
    session_id: Optional[str] = Field(None, description="Chat session ID for conversation continuity")


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Chat session ID")
    conversation_count: int = Field(..., description="Number of messages in this conversation")


class ConversationSession(BaseModel):
    """Base chat conversation session"""
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Conversation messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field("healthy", description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    openai_configured: bool = Field(..., description="Whether OpenAI API is properly configured")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class V2ConversationSession(ConversationSession):
    """
    V2 conversation session with pandas DataFrames for direct log file analysis.
    """
    dataframes: Dict[str, Any] = Field(default_factory=dict, description="Pandas DataFrames for analysis")
    dataframe_schemas: Dict[str, Any] = Field(default_factory=dict, description="Schemas of the DataFrames")
    is_processing: bool = Field(False, description="Flag to indicate if a log file is currently being processed")
    processing_error: Optional[str] = Field(None, description="Stores any error message from the processing stage")

    class Config:
        arbitrary_types_allowed = True 