"""
Pydantic models for UAV Log Analyzer Chat API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")


class TelemetryMessage(BaseModel):
    """Individual telemetry message type (e.g., GPS, ATT, MODE)"""
    time_boot_ms: Optional[List[float]] = Field(None, description="Timestamps in milliseconds")
    # GPS fields
    lat: Optional[List[float]] = Field(None, description="Latitude values")
    lon: Optional[List[float]] = Field(None, description="Longitude values") 
    lng: Optional[List[float]] = Field(None, description="Longitude values (alternative)")
    alt: Optional[List[float]] = Field(None, description="Altitude values")
    # Attitude fields
    Roll: Optional[List[float]] = Field(None, description="Roll values in degrees")
    Pitch: Optional[List[float]] = Field(None, description="Pitch values in degrees")
    Yaw: Optional[List[float]] = Field(None, description="Yaw values in degrees")
    # Mode fields
    Mode: Optional[List[int]] = Field(None, description="Flight mode numbers")
    asText: Optional[List[str]] = Field(None, description="Flight mode text descriptions")
    # Battery/Power fields
    Volt: Optional[List[float]] = Field(None, description="Battery voltage")
    Curr: Optional[List[float]] = Field(None, description="Current draw")
    CurrTot: Optional[List[float]] = Field(None, description="Total current consumed")
    # Allow additional fields from ArduPilot messages
    class Config:
        extra = "allow"


class TelemetryMetadata(BaseModel):
    """Metadata about the telemetry data"""
    startTime: Optional[int] = Field(None, description="Start time in milliseconds since epoch")
    vehicleType: Optional[str] = Field(None, description="Type of vehicle")
    logType: Optional[str] = Field(None, description="Log file type (bin, tlog, txt)")
    duration: Optional[float] = Field(None, description="Flight duration in seconds")
    messageCount: Optional[int] = Field(None, description="Total number of messages")


class TelemetryData(BaseModel):
    """Complete telemetry data structure matching frontend format"""
    messages: Dict[str, TelemetryMessage] = Field(..., description="Telemetry messages by type")
    metadata: Optional[TelemetryMetadata] = Field(None, description="Telemetry metadata")
    availableMessages: Optional[Dict[str, Any]] = Field(None, description="Available message types")


class ChatRequest(BaseModel):
    """Request to chat endpoint with telemetry context"""
    message: str = Field(..., description="User's question about the telemetry data")
    session_id: Optional[str] = Field(None, description="Chat session ID for conversation continuity")
    telemetry_data: Optional[TelemetryData] = Field(None, description="Parsed telemetry data from frontend")
    clear_history: bool = Field(False, description="Whether to clear conversation history")


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Chat session ID")
    conversation_count: int = Field(..., description="Number of messages in this conversation")


class ConversationSession(BaseModel):
    """Chat conversation session"""
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Conversation messages")
    telemetry_data: Optional[TelemetryData] = Field(None, description="Associated telemetry data")
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