"""
Chat Service Module

Handles OpenAI API interactions and manages conversation context for the UAV log chatbot.
"""

from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import uuid

from models import ChatMessage, ConversationSession, TelemetryData
from config import settings
from telemetry_parser import TelemetryParser

class ChatService:
    """
    Service class for managing chatbot conversations and OpenAI API interactions.
    """
    
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            print("WARNING: OpenAI API key not configured. Chat functionality will be limited.")
            self.client = None
        else:
            try:
                self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception as e:
                print(f"ERROR: Failed to initialize OpenAI client: {e}")
                self.client = None
        self.sessions: Dict[str, ConversationSession] = {}
        self.telemetry_parser = TelemetryParser()
        
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = ConversationSession(
            session_id=session_id,
            messages=[],
            telemetry_data=None
        )
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing conversation session."""
        return self.sessions.get(session_id)
    
    def update_session_telemetry(self, session_id: str, telemetry_data: Dict[str, Any]):
        """Update the telemetry data for a session."""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        # Parse the telemetry data
        self.telemetry_parser.messages = telemetry_data.get('messages', {})
        self.telemetry_parser.metadata = telemetry_data.get('metadata', {})
        
        # Store in session
        self.sessions[session_id].telemetry_data = TelemetryData(
            messages=telemetry_data.get('messages', {}),
            metadata=telemetry_data.get('metadata', {}),
            start_time=datetime.fromtimestamp(telemetry_data.get('metadata', {}).get('startTime', 0) / 1000) if telemetry_data.get('metadata', {}).get('startTime') else None
        )
        self.sessions[session_id].last_activity = datetime.now()
    
    def _get_system_prompt(self, telemetry_available: bool = False) -> str:
        """Get the system prompt for the chatbot."""
        base_prompt = """You are a specialized UAV/drone flight log analyst chatbot. You help users analyze and understand MAVLink telemetry data from drone flight logs (.bin files).

Your capabilities include:
- Analyzing flight telemetry data including GPS tracks, altitude, battery status, flight modes, and sensor readings
- Answering questions about flight performance, safety events, and flight parameters
- Providing insights about flight patterns, anomalies, and system status
- Explaining MAVLink message types and their significance

Guidelines:
- Be precise and technical when discussing telemetry data
- Always provide context for your answers (timestamps, units, etc.)
- If you need clarification about what specific data the user wants, ask for it
- Reference the ArduPilot documentation when relevant: https://ardupilot.org/plane/docs/logmessages.html
- Format numerical data clearly with appropriate units
- Highlight any safety-critical findings or anomalies"""

        if telemetry_available:
            base_prompt += """

IMPORTANT: You have access to parsed telemetry data from the user's flight log. When answering questions:
- Use specific data from the telemetry when available
- Provide actual timestamps, values, and measurements
- Reference specific message types (GPS, ATT, MODE, etc.) when relevant
- If the requested data isn't available in the current log, explain what's missing"""
        else:
            base_prompt += """

NOTE: No telemetry data has been loaded yet. Ask the user to upload a flight log file (.bin, .tlog) to begin analysis."""

        return base_prompt
    
    def _format_telemetry_context(self, query: str, session: ConversationSession) -> str:
        """Format telemetry data as context for the LLM."""
        if not session.telemetry_data:
            return "No telemetry data available."
        
        # Query relevant telemetry data based on the user's question
        query_result = self.telemetry_parser.query_telemetry_data(query)
        
        context_parts = [
            f"=== TELEMETRY DATA CONTEXT ===",
            f"Query: {query}",
            f"Available message types: {list(session.telemetry_data.messages.keys())}"
        ]
        
        # Add relevant data based on query
        if query_result.get("data"):
            context_parts.append("=== RELEVANT DATA ===")
            for data_type, data_content in query_result["data"].items():
                context_parts.append(f"{data_type.upper()}: {json.dumps(data_content, indent=2)}")
        
        # Add flight summary for general queries
        if not query_result.get("data") or len(query_result["data"]) == 1 and "summary" in query_result["data"]:
            summary = self.telemetry_parser.get_data_summary()
            context_parts.extend([
                "=== FLIGHT SUMMARY ===",
                f"Total message types: {len(summary.get('message_types', []))}",
                f"Total messages: {summary.get('total_messages', 0)}",
                f"Flight statistics: {json.dumps(summary.get('flight_statistics', {}), indent=2)}"
            ])
        
        context_parts.append("=== END TELEMETRY CONTEXT ===")
        
        return "\n".join(context_parts)
    
    async def get_chat_response(self, session_id: str, user_message: str) -> str:
        """Get a response from the chatbot for the given message."""
        # Check if OpenAI client is available
        if not self.client:
            return "I apologize, but the OpenAI API is not configured. Please set up your OpenAI API key in the .env file to enable chat functionality."
        
        # Ensure session exists
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        
        # Add user message to conversation history
        user_msg = ChatMessage(role="user", content=user_message)
        session.messages.append(user_msg)
        session.last_activity = datetime.now()
        
        # Prepare messages for OpenAI API
        messages = [
            {"role": "system", "content": self._get_system_prompt(session.telemetry_data is not None)}
        ]
        
        # Add telemetry context if available
        if session.telemetry_data:
            telemetry_context = self._format_telemetry_context(user_message, session)
            messages.append({"role": "system", "content": telemetry_context})
        
        # Add conversation history (keep last 10 exchanges to avoid token limits)
        recent_messages = session.messages[-20:]  # Last 20 messages (10 exchanges)
        for msg in recent_messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to conversation history
            assistant_msg = ChatMessage(role="assistant", content=assistant_response)
            session.messages.append(assistant_msg)
            
            return assistant_response
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            
            # Still add the error response to conversation history
            assistant_msg = ChatMessage(role="assistant", content=error_message)
            session.messages.append(assistant_msg)
            
            return error_message
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the conversation history for a session."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in session.messages
        ]
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "has_telemetry_data": session.telemetry_data is not None,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        } 