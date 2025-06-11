"""
AI Chat Service for UAV Log Analysis

This service provides OpenAI-powered analysis of telemetry data parsed by the frontend.
It includes comprehensive knowledge of ArduPilot log messages and flight data analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from openai import AsyncOpenAI
from config import get_settings
from models import (
    ChatMessage, 
    ConversationSession, 
    TelemetryData,
    TelemetryMessage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    """AI-powered chat service for UAV telemetry analysis"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.openai_api_key
        )
        self.sessions: Dict[str, ConversationSession] = {}
        
        # ArduPilot message documentation (from https://ardupilot.org/plane/docs/logmessages.html)
        self.ardupilot_messages = {
            "ACC": "IMU accelerometer data - contains AccX, AccY, AccZ acceleration values",
            "ADSB": "Automatic Dependent Surveillance-Broadcast detected vehicle information",
            "AETR": "Normalised pre-mixer control surface outputs - Aileron, Elevator, Throttle, Rudder",
            "AHR2": "Backup AHRS data - Roll, Pitch, Yaw, Alt, Lat, Lng and quaternion components",
            "ATT": "Attitude data - Roll, Pitch, Yaw from the attitude controller",
            "BARO": "Barometer data - Alt (altitude), Press (pressure), Temp (temperature)",
            "CURR": "Battery/Power data - Volt (voltage), Curr (current), CurrTot (total current)",
            "GPS": "GPS position data - Lat, Lng, Alt, Spd (speed), GCrs (ground course)",
            "GPS2": "Secondary GPS data when multiple GPS units are present",
            "IMU": "Inertial measurement unit data - AccX, AccY, AccZ, GyrX, GyrY, GyrZ",
            "MODE": "Flight mode changes - Mode number and asText description",
            "MSG": "Text messages and alerts from the autopilot system",
            "PARM": "Parameter values set in the autopilot",
            "POS": "Position estimates from the EKF - Lat, Lng, Alt",
            "RCIN": "RC input values from transmitter channels",
            "RCOU": "RC output values to servos and motors",
            "VIBE": "Vibration levels affecting IMU performance",
            "XKF1": "Extended Kalman Filter states and innovations",
            "XKF2": "More EKF data including wind estimates",
            "XKF3": "EKF innovation variances and health monitoring",
            "XKF4": "EKF timing and processing information"
        }
        
    def _get_system_prompt(self, telemetry_data: Optional[TelemetryData] = None) -> str:
        """Generate system prompt with telemetry context and ArduPilot knowledge"""
        
        base_prompt = """You are an expert UAV flight data analyst specializing in ArduPilot/PX4 telemetry analysis. 
        You help pilots and engineers understand flight logs, diagnose issues, and improve flight performance.

        You have comprehensive knowledge of ArduPilot log messages including:
        """
        
        # Add ArduPilot message knowledge
        for msg_type, description in self.ardupilot_messages.items():
            base_prompt += f"\n- {msg_type}: {description}"
            
        base_prompt += """

        CAPABILITIES:
        - Analyze flight performance and behavior patterns
        - Identify potential issues or anomalies in flight data
        - Explain flight modes and transitions
        - Assess GPS, attitude, and sensor data quality
        - Evaluate battery performance and power consumption
        - Detect vibration issues and mechanical problems
        - Review parameter settings and their effects
        - Provide flight improvement recommendations

        ANALYSIS APPROACH:
        1. Always consider the complete flight context
        2. Look for correlations between different data streams
        3. Identify patterns that might indicate issues
        4. Provide specific, actionable insights
        5. Ask clarifying questions when more context is needed
        6. Reference specific time periods or data points in your analysis
        """
        
        # Add telemetry context if available
        if telemetry_data and telemetry_data.messages:
            base_prompt += "\n\nCURRENT FLIGHT DATA CONTEXT:\n"
            
            # Add metadata context
            if telemetry_data.metadata:
                metadata = telemetry_data.metadata
                if metadata.logType:
                    base_prompt += f"- Log Type: {metadata.logType}\n"
                if metadata.vehicleType:
                    base_prompt += f"- Vehicle Type: {metadata.vehicleType}\n"
                if metadata.startTime:
                    start_time = datetime.fromtimestamp(metadata.startTime / 1000)
                    base_prompt += f"- Flight Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Add message type summary
            base_prompt += f"- Available Message Types: {', '.join(telemetry_data.messages.keys())}\n"
            
            # Add data summary for key message types
            for msg_type, msg_data in telemetry_data.messages.items():
                if hasattr(msg_data, 'time_boot_ms') and msg_data.time_boot_ms:
                    duration = (max(msg_data.time_boot_ms) - min(msg_data.time_boot_ms)) / 1000
                    count = len(msg_data.time_boot_ms)
                    base_prompt += f"- {msg_type}: {count} messages over {duration:.1f} seconds\n"
            
        base_prompt += """
        
        Provide helpful, accurate analysis focusing on flight safety and performance optimization.
        Always explain your reasoning and reference specific data when making assessments.
        """
        
        return base_prompt
        
    def _format_telemetry_context(self, telemetry_data: TelemetryData) -> str:
        """Format telemetry data for inclusion in chat context"""
        
        if not telemetry_data or not telemetry_data.messages:
            return "No telemetry data available."
            
        context = "TELEMETRY DATA SUMMARY:\n\n"
        
        # Add metadata
        if telemetry_data.metadata:
            context += "FLIGHT METADATA:\n"
            metadata = telemetry_data.metadata
            if metadata.startTime:
                start_time = datetime.fromtimestamp(metadata.startTime / 1000)
                context += f"- Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            if metadata.logType:
                context += f"- Log Type: {metadata.logType}\n"
            if metadata.vehicleType:
                context += f"- Vehicle: {metadata.vehicleType}\n"
            context += "\n"
        
        # Add message summaries
        context += "MESSAGE SUMMARY:\n"
        
        for msg_type, msg_data in telemetry_data.messages.items():
            if not hasattr(msg_data, 'time_boot_ms') or not msg_data.time_boot_ms:
                continue
                
            time_data = msg_data.time_boot_ms
            start_time = min(time_data) / 1000
            end_time = max(time_data) / 1000
            duration = end_time - start_time
            count = len(time_data)
            
            context += f"- {msg_type}: {count} messages ({duration:.1f}s duration)\n"
            
            # Add specific insights for key message types
            if msg_type == "GPS" and hasattr(msg_data, 'lat') and msg_data.lat:
                lat_range = max(msg_data.lat) - min(msg_data.lat)
                lon_range = max(msg_data.lon or msg_data.lng or [0]) - min(msg_data.lon or msg_data.lng or [0])
                context += f"  GPS range: {lat_range:.6f}° lat, {lon_range:.6f}° lon\n"
                
            elif msg_type == "ATT" and hasattr(msg_data, 'Roll') and msg_data.Roll:
                max_roll = max(abs(r) for r in msg_data.Roll)
                max_pitch = max(abs(p) for p in msg_data.Pitch) if msg_data.Pitch else 0
                context += f"  Max attitudes: {max_roll:.1f}° roll, {max_pitch:.1f}° pitch\n"
                
            elif msg_type == "MODE" and hasattr(msg_data, 'asText') and msg_data.asText:
                unique_modes = list(set(msg_data.asText))
                context += f"  Flight modes: {', '.join(unique_modes)}\n"
                
            elif msg_type == "CURR" and hasattr(msg_data, 'Volt') and msg_data.Volt:
                min_volt = min(msg_data.Volt)
                max_volt = max(msg_data.Volt)
                context += f"  Battery: {min_volt:.1f}V - {max_volt:.1f}V\n"
        
        return context
        
    async def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new chat session"""
        if not session_id:
            session_id = str(uuid.uuid4())
            
        self.sessions[session_id] = ConversationSession(
            session_id=session_id,
            messages=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        logger.info(f"Created new chat session: {session_id}")
        return session_id
        
    async def update_session_telemetry(self, session_id: str, telemetry_data: TelemetryData):
        """Update session with new telemetry data"""
        if session_id not in self.sessions:
            await self.create_session(session_id)
            
        self.sessions[session_id].telemetry_data = telemetry_data
        self.sessions[session_id].last_updated = datetime.now()
        
        logger.info(f"Updated telemetry data for session {session_id}")
        
    async def clear_session_history(self, session_id: str):
        """Clear conversation history but keep telemetry data"""
        if session_id in self.sessions:
            telemetry_data = self.sessions[session_id].telemetry_data
            self.sessions[session_id].messages = []
            self.sessions[session_id].telemetry_data = telemetry_data
            self.sessions[session_id].last_updated = datetime.now()
            
    async def chat(self, session_id: str, user_message: str) -> str:
        """Process a chat message and return AI response"""
        
        if session_id not in self.sessions:
            await self.create_session(session_id)
            
        session = self.sessions[session_id]
        
        # Add user message to session
        session.messages.append(ChatMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now()
        ))
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": self._get_system_prompt(session.telemetry_data)}
        ]
        
        # Add telemetry context if available
        if session.telemetry_data:
            telemetry_context = self._format_telemetry_context(session.telemetry_data)
            messages.append({
                "role": "system", 
                "content": f"Current flight data:\n{telemetry_context}"
            })
            
        # Add conversation history (limit to recent messages to stay within token limits)
        recent_messages = session.messages[-10:]  # Keep last 10 messages
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
            
        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to session
            session.messages.append(ChatMessage(
                role="assistant",
                content=assistant_response,
                timestamp=datetime.now()
            ))
            
            session.last_updated = datetime.now()
            
            logger.info(f"Generated response for session {session_id}")
            return assistant_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            error_response = "I'm having trouble analyzing your flight data right now. Please check that the OpenAI API is properly configured and try again."
            
            session.messages.append(ChatMessage(
                role="assistant",
                content=error_response,
                timestamp=datetime.now()
            ))
            
            return error_response
            
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
        
    def cleanup_old_sessions(self, hours: int = 24):
        """Remove sessions older than specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        old_sessions = [
            sid for sid, session in self.sessions.items()
            if session.last_updated < cutoff
        ]
        
        for session_id in old_sessions:
            del self.sessions[session_id]
            
        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old sessions")


# Global chat service instance
chat_service = ChatService() 