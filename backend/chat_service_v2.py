"""
AI Chat Service V2 with Enhanced Agentic Tooling for UAV Log Analysis

This service introduces an enhanced agentic approach to telemetry data analysis.
It dynamically fetches documentation, provides multiple analysis tools, and delivers
user-friendly responses for complex UAV flight data questions.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import json
import pandas as pd
import numpy as np
from io import StringIO
import traceback
import requests
import httpx
import asyncio
from bs4 import BeautifulSoup
import re
import time
from functools import lru_cache

from openai import OpenAI
from pymavlink import DFReader
from config import get_settings
from models import (
    ChatMessage,
    ConversationSession, # Reusing this for session state
    TelemetryData,
    V2ConversationSession,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatServiceV2:
    """
    An enhanced agentic chat service that uses OpenAI's tool-calling feature to
    generate and execute Python code for telemetry data analysis with dynamic
    documentation and multiple analysis tools.
    """

    def __init__(self):
        """
        Initializes the service, loading settings and setting up the OpenAI client.
        """
        self.settings = get_settings()
        # Using the synchronous client here for simplicity in the exec environment.
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured. V2 service will not be functional.")
        self.sessions: Dict[str, V2ConversationSession] = {}
        self.http_client = httpx.AsyncClient(timeout=10)
        
        # Static fallback ArduPilot message documentation
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
        
        # Cache for dynamic documentation
        self.documentation_cache = {}
        self.last_doc_fetch = {}

    # @lru_cache is not compatible with async methods. The manual cache implementation is used instead.
    async def _check_url_alive(self, url: str) -> bool:
        """
        Check if a URL is alive and reachable with a quick HEAD request.
        """
        try:
            response = await self.http_client.head(url, timeout=5)
            return response.status_code < 400
        except Exception:
            return False

    async def _fetch_ardupilot_documentation(self, message_type: str) -> str:
        """
        Dynamically fetch ArduPilot documentation for specific message types.
        Returns cached static documentation as fallback.
        First checks if the URL is alive before attempting to fetch.
        """
        try:
            # Try to fetch from ArduPilot docs website
            url = f"https://ardupilot.org/plane/docs/logmessages.html"
            
            # Check cache first (cache for 1 hour) 
            cache_key = f"{message_type}_{url}"
            if (cache_key in self.documentation_cache and 
                cache_key in self.last_doc_fetch and 
                time.time() - self.last_doc_fetch[cache_key] < 3600):
                return self.documentation_cache[cache_key]
            
            # Check if URL is alive before making the full request
            if not await self._check_url_alive(url):
                logger.debug(f"ArduPilot documentation URL is not reachable, skipping fetch for {message_type}")
                # Fallback to static documentation immediately
                return self.ardupilot_messages.get(message_type, f"Data related to {message_type} from the flight log.")
            
            response = await self.http_client.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for message type documentation
                text = soup.get_text()
                pattern = rf'{message_type}[:\s]+(.*?)(?=\n[A-Z][A-Z][A-Z0-9]*[:\s]|$)'
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                
                if match:
                    doc = match.group(1).strip()[:500]  # Limit length
                    self.documentation_cache[cache_key] = doc
                    self.last_doc_fetch[cache_key] = time.time()
                    return doc
            
        except Exception as e:
            logger.debug(f"Failed to fetch dynamic documentation for {message_type}: {e}")
        
        # Fallback to static documentation
        return self.ardupilot_messages.get(message_type, f"Data related to {message_type} from the flight log.")

    def _get_data_summary(self, session: V2ConversationSession) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the available data for better context.
        """
        summary = {
            "message_types": len(session.dataframes),
            "total_records": sum(len(df) for df in session.dataframes.values()),
            "time_range": None,
            "key_metrics": {}
        }
        
        try:
            # Find time range
            timestamps = []
            for df in session.dataframes.values():
                if 'timestamp' in df.columns:
                    timestamps.extend(df['timestamp'].dropna().tolist())
            
            if timestamps:
                min_time = min(timestamps)
                max_time = max(timestamps)
                summary["time_range"] = {
                    "start": min_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": max_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_minutes": (max_time - min_time).total_seconds() / 60
                }
            
            # Key metrics
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Alt' in gps_df.columns:
                    summary["key_metrics"]["max_altitude"] = float(gps_df['Alt'].max())
                if 'Spd' in gps_df.columns:
                    summary["key_metrics"]["max_speed"] = float(gps_df['Spd'].max())
            
            if 'BARO' in session.dataframes:
                baro_df = session.dataframes['BARO']
                if 'Alt' in baro_df.columns:
                    summary["key_metrics"]["max_baro_altitude"] = float(baro_df['Alt'].max())
            
        except Exception as e:
            logger.debug(f"Error generating data summary: {e}")
        
        return summary

    async def _get_system_prompt(self, session: V2ConversationSession) -> str:
        """
        Generates a concise system prompt for precise responses.
        """
        if not session.dataframes:
            return "You are a flight data analysis assistant. Give brief, accurate answers."

        data_summary = self._get_data_summary(session)
        
        prompt = f"""You are a UAV flight data analyst. Answer questions precisely and concisely.

DATA: {data_summary['message_types']} data types, {data_summary['total_records']} records."""

        if data_summary.get('time_range'):
            prompt += f"""
Flight: {data_summary['time_range']['duration_minutes']:.1f} min ({data_summary['time_range']['start']} to {data_summary['time_range']['end']})"""

        if data_summary.get('key_metrics'):
            prompt += f"\nMetrics: {data_summary['key_metrics']}"

        prompt += """

DATAFRAMES:
"""

        # Limit to most important dataframes
        important_dataframes = ['GPS', 'BARO', 'ATT', 'MODE', 'CURR', 'IMU', 'RCIN', 'RCOU', 'MSG']
        dataframes_to_show = []
        
        for name in important_dataframes:
            if name in session.dataframe_schemas:
                dataframes_to_show.append(name)
        
        for name in session.dataframe_schemas.keys():
            if name not in dataframes_to_show and len(dataframes_to_show) < 10:
                dataframes_to_show.append(name)

        # Fetch documentation concurrently
        doc_coroutines = [self._fetch_ardupilot_documentation(name) for name in dataframes_to_show]
        docs = await asyncio.gather(*doc_coroutines)
        doc_map = {name: doc for name, doc in zip(dataframes_to_show, docs)}

        for name in dataframes_to_show:
            if name in session.dataframe_schemas:
                schema = session.dataframe_schemas[name]
                doc = doc_map.get(name, f"{name} flight data.")
                # Truncate documentation to first sentence only
                doc_short = doc.split('.')[0] + '.' if '.' in doc else doc[:50]
                prompt += f"\n- `dfs['{name}']` ({len(session.dataframes[name])} records): {doc_short}\n"
                # Show only key columns
                columns = list(schema['columns'].keys())[:5]
                prompt += f"  Columns: {columns}\n"
        
        if len(session.dataframes) > len(dataframes_to_show):
            prompt += f"\n+ {len(session.dataframes) - len(dataframes_to_show)} more dataframes\n"

        prompt += """
            AVAILABLE TOOLS:
            - execute_python_code: For calculations and custom analysis
            - find_anomalies: Enhanced anomaly detection with severity levels and timestamps
            - detect_flight_events: Find GPS loss, mode changes, critical alerts with exact timestamps
            - analyze_flight_phase: Detailed takeoff/cruise/landing analysis
            - get_timeline_analysis: Chronological timeline of all flight events
            - compare_metrics: Compare different flight parameters
            - generate_insights: Flight performance and safety summary

            RULES:
            - Always provide specific timestamps for events (use detect_flight_events)
            - Categorize issues by severity (Critical/Warning/Info)
            - Use emojis and clear formatting for better readability
            - For GPS questions, use detect_flight_events with 'gps_loss' type
            - For error analysis, use find_anomalies with appropriate focus areas
            - Include context about what anomalies mean for flight safety"""
        return prompt

    def _make_response_user_friendly(self, technical_response: str, context: str = "") -> str:
        """
        Convert technical responses to be more concise and precise.
        """
        try:
            # Skip conversion if response is already short or contains errors
            if len(technical_response) < 30 or any(word in technical_response.lower() for word in ['sorry', 'help', 'error', 'try again']):
                return technical_response
            
            # Make the response more concise and precise
            friendly_prompt = f"""Make this response concise and precise. Remove unnecessary explanations.

            Original: {technical_response}

            Requirements:
            - Keep only essential information
            - Use specific numbers
            - Remove verbose explanations
            - Be direct and factual

            Concise response:"""

            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[{"role": "user", "content": friendly_prompt}],
                max_tokens=200,
                timeout=15
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to make response concise: {e}", exc_info=True)
            return technical_response

    async def create_or_get_session(self, session_id: Optional[str] = None) -> V2ConversationSession:
        """Creates a new V2 session or retrieves an existing one."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = V2ConversationSession(session_id=session_id)
            logger.info(f"Created new V2 chat session: {session_id}")
        
        return self.sessions[session_id]

    async def process_log_file(self, session_id: str, file_path: str):
        """
        Parses a log file (e.g., .bin) directly into pandas DataFrames
        and stores them in the session. This is designed to be run as a
        background task.
        """
        session = await self.create_or_get_session(session_id)
        logger.info(f"[{session_id}] Starting log file processing for {file_path}")

        try:
            session.dataframes = {}
            session.dataframe_schemas = {}
            session.processing_error = None

            logger.info(f"[{session_id}] Opening log file with DFReader: {file_path}")
            log = DFReader.DFReader_binary(file_path)
            
            data = {}
            message_count = 0
            while True:
                msg = log.recv_msg()
                if msg is None:
                    break
                msg_type = msg.get_type()
                if msg_type not in data:
                    data[msg_type] = []
                data[msg_type].append(msg.to_dict())
                message_count += 1
                
                # Log progress every 1000 messages
                if message_count % 1000 == 0:
                    logger.info(f"[{session_id}] Processed {message_count} messages...")

            logger.info(f"[{session_id}] Finished reading {message_count} messages, found {len(data)} message types: {list(data.keys())}")

            for msg_type, msg_list in data.items():
                if not msg_list:
                    continue
                
                logger.info(f"[{session_id}] Creating DataFrame for {msg_type} with {len(msg_list)} messages")
                df = pd.DataFrame(msg_list)
                
                if 'TimeUS' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['TimeUS'], unit='us')
                elif 'time_boot_ms' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time_boot_ms'], unit='ms')

                session.dataframes[msg_type] = df
                
                session.dataframe_schemas[msg_type] = {
                    "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "description": f"Contains {msg_type} data from the flight log."
                }
            
            logger.info(f"[{session_id}] Successfully processed and created DataFrames for {len(session.dataframes)} message types: {list(session.dataframes.keys())}")

        except Exception as e:
            session.processing_error = str(e)
            logger.error(f"[{session_id}] Error processing log file {file_path}: {e}", exc_info=True)
            
        finally:
            session.is_processing = False
            session.last_updated = datetime.now()

    async def chat(self, session_id: str, user_message: str) -> str:
        """
        Enhanced chat method with multiple tools and user-friendly responses.
        """
        logger.info(f"[{session_id}] Chat request received: '{user_message[:50]}...'")
        
        if not self.client:
            logger.error("OpenAI client is not configured")
            return "AI service unavailable. Check configuration."

        session = await self.create_or_get_session(session_id)
        
        logger.info(f"[{session_id}] Session status - Processing: {session.is_processing}, DataFrames: {len(session.dataframes)}, Error: {session.processing_error}")

        if session.is_processing:
            logger.info(f"[{session_id}] Session still processing, returning wait message")
            return "Still processing log. Please wait."

        if session.processing_error:
            logger.error(f"[{session_id}] Processing error found: {session.processing_error}")
            return f"Log processing error: {session.processing_error}"

        if not session.dataframes:
            logger.warning(f"[{session_id}] No dataframes found in session")
            return "No flight data loaded. Please upload a log file first."

        session.messages.append(ChatMessage(role="user", content=user_message))

        try:
            logger.info(f"[{session_id}] Starting chat processing with {len(session.dataframes)} dataframes")
            # Enhanced tools for better analysis
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_python_code",
                        "description": "Execute Python code to analyze flight data and calculate specific metrics.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Python code to execute for data analysis.",
                                }
                            },
                            "required": ["code"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "find_anomalies",
                        "description": "Detect unusual patterns, errors, or anomalies in the flight data.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "focus_areas": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific areas to check for anomalies (e.g., 'GPS', 'altitude', 'vibration').",
                                }
                            },
                            "required": ["focus_areas"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "compare_metrics",
                        "description": "Compare different flight parameters or time periods.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "metrics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Metrics to compare (e.g., 'GPS_altitude', 'BARO_altitude').",
                                },
                                "comparison_type": {
                                    "type": "string",
                                    "description": "Type of comparison ('correlation', 'difference', 'trend').",
                                }
                            },
                            "required": ["metrics", "comparison_type"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "generate_insights",
                        "description": "Generate comprehensive insights and summary of the flight.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "focus": {
                                    "type": "string",
                                    "description": "What to focus on ('overall', 'performance', 'safety', 'efficiency').",
                                }
                            },
                            "required": ["focus"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "detect_flight_events",
                        "description": "Detect specific flight events like GPS loss, mode changes, critical alerts with timestamps.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "event_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Types of events to detect: 'gps_loss', 'mode_changes', 'critical_alerts', 'power_issues', 'attitude_problems'",
                                }
                            },
                            "required": ["event_types"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_flight_phase",
                        "description": "Analyze specific phases of flight (takeoff, cruise, landing) with detailed metrics.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "phase": {
                                    "type": "string",
                                    "description": "Flight phase to analyze: 'takeoff', 'cruise', 'landing', 'all'",
                                },
                                "metrics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific metrics to focus on: 'altitude', 'speed', 'power', 'stability'",
                                }
                            },
                            "required": ["phase"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_timeline_analysis",
                        "description": "Provide a chronological timeline of key events and issues during the flight.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time_resolution": {
                                    "type": "string",
                                    "description": "Time resolution for timeline: 'seconds', 'minutes', 'auto'",
                                }
                            },
                            "required": ["time_resolution"],
                        },
                    },
                }
            ]

            # Call the model with enhanced tools
            system_prompt = await self._get_system_prompt(session)
            logger.info(f"[{session_id}] System prompt length: {len(system_prompt)} characters")
            
            # If prompt is too long, use a simplified version
            if len(system_prompt) > 15000:
                logger.warning(f"[{session_id}] System prompt is very long ({len(system_prompt)} chars), using simplified version")
                system_prompt = f"""You are a UAV flight data analyst. Answer questions precisely and concisely.

DATA: {len(session.dataframes)} data types, {len(session.dataframes)} records."""
            
            logger.info(f"[{session_id}] Calling OpenAI API with tools (model: {self.settings.openai_model})")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    tools=tools,
                    tool_choice="auto",
                    timeout=30
                )
                logger.info(f"[{session_id}] Received response from OpenAI API")
            except Exception as api_error:
                logger.error(f"[{session_id}] OpenAI API call failed: {api_error}", exc_info=True)
                # Try a simplified approach without tools
                logger.info(f"[{session_id}] Retrying without tools")
                try:
                    response = self.client.chat.completions.create(
                        model=self.settings.openai_model,
                        messages=[
                            {"role": "system", "content": "You are a flight data analysis assistant. Give brief, accurate answers."},
                            {"role": "user", "content": f"The user asked: {user_message}. The flight data contains {len(session.dataframes)} different types of data including GPS, altitude, and other flight parameters. Please provide a helpful response."}
                        ],
                        timeout=30,
                        max_tokens=500
                    )
                    logger.info(f"[{session_id}] Fallback response received")
                    return response.choices[0].message.content or "Analysis failed. System may be overloaded."
                except Exception as fallback_error:
                    logger.error(f"[{session_id}] Fallback API call also failed: {fallback_error}")
                    return "Technical difficulties. Please try again."

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if not tool_calls:
                # If no tool was called, provide a direct response
                logger.info(f"[{session_id}] No tools called, providing direct response")
                answer = response_message.content or "Unable to process question. Please be more specific."
                session.messages.append(ChatMessage(role="assistant", content=answer))
                logger.info(f"[{session_id}] Making response user-friendly")
                user_friendly_answer = self._make_response_user_friendly(answer)
                logger.info(f"[{session_id}] Returning user-friendly response")
                return user_friendly_answer

            # Process tool calls
            tool_results = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"[{session_id}] Executing tool: {function_name}")
                
                if function_name == "execute_python_code":
                    result = self._execute_python_code(session, function_args["code"])
                elif function_name == "find_anomalies":
                    result = self._find_anomalies(session, function_args["focus_areas"])
                elif function_name == "compare_metrics":
                    result = self._compare_metrics(session, function_args["metrics"], function_args["comparison_type"])
                elif function_name == "generate_insights":
                    result = self._generate_insights(session, function_args["focus"])
                elif function_name == "detect_flight_events":
                    result = self._detect_flight_events(session, function_args["event_types"])
                elif function_name == "analyze_flight_phase":
                    result = self._analyze_flight_phase(session, function_args["phase"], function_args["metrics"])
                elif function_name == "get_timeline_analysis":
                    result = self._get_timeline_analysis(session, function_args["time_resolution"])
                else:
                    result = f"Unknown tool: {function_name}"
                
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(result),
                })

            # Get final response from the model
            logger.info(f"[{session_id}] Getting final response from OpenAI after tool execution")
            final_response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    response_message,
                ] + tool_results,
                timeout=30
            )
            
            final_answer = final_response.choices[0].message.content
            logger.info(f"[{session_id}] Got final answer from OpenAI")
            
            # Make the response more user-friendly
            logger.info(f"[{session_id}] Making final response user-friendly")
            user_friendly_answer = self._make_response_user_friendly(final_answer, user_message)
            logger.info(f"[{session_id}] User-friendly conversion complete")
            
            session.messages.append(ChatMessage(role="assistant", content=user_friendly_answer))
            logger.info(f"[{session_id}] Chat processing completed successfully")
            return user_friendly_answer

        except Exception as e:
            logger.error(f"Error in V2 chat processing for session {session_id}: {e}", exc_info=True)
            return "Unexpected error in analysis. Please try again or rephrase."

    def _execute_python_code(self, session: V2ConversationSession, code: str) -> str:
        """Execute Python code with enhanced error handling and user-friendly output."""
        try:
            logger.info(f"Executing AI-generated code for session {session.session_id}:\n{code}")
            
            local_scope = {"dfs": session.dataframes, "pd": pd, "np": np}
            
            code_lines = code.strip().split('\n')
            
            if len(code_lines) > 1:
                exec('\n'.join(code_lines[:-1]), {}, local_scope)
            
            result = eval(code_lines[-1], {}, local_scope)
            return f"Analysis successful. Result: {str(result)}"

        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return f"Analysis error: {str(e)}"

    def _find_anomalies(self, session: V2ConversationSession, focus_areas: List[str]) -> str:
        """Enhanced anomaly detection with temporal analysis and severity assessment."""
        critical_issues = []
        warning_issues = []
        info_issues = []
        
        try:
            for area in focus_areas:
                if area.upper() in session.dataframes:
                    df = session.dataframes[area.upper()]
                    
                    # GPS-specific analysis
                    if area.upper() == 'GPS':
                        issues = self._analyze_gps_anomalies(df)
                        critical_issues.extend(issues['critical'])
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
                    
                    # Attitude control analysis
                    elif area.upper() == 'ATT':
                        issues = self._analyze_attitude_anomalies(df)
                        critical_issues.extend(issues['critical'])
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
                    
                    # Power system analysis
                    elif area.upper() == 'CURR':
                        issues = self._analyze_power_anomalies(df)
                        critical_issues.extend(issues['critical'])
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
                    
                    # General analysis for other systems
                    else:
                        issues = self._analyze_general_anomalies(df, area.upper())
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
            
            # Format response with prioritization
            response_parts = []
            
            if critical_issues:
                response_parts.append("CRITICAL ISSUES:")
                for issue in critical_issues:
                    response_parts.append(f"  • {issue}")
            
            if warning_issues:
                response_parts.append("\nWARNING ISSUES:")
                for issue in warning_issues:
                    response_parts.append(f"  • {issue}")
            
            if info_issues:
                response_parts.append("\nINFO:")
                for issue in info_issues:
                    response_parts.append(f"  • {issue}")
            
            if not (critical_issues or warning_issues or info_issues):
                return "No significant anomalies detected in the analyzed areas."
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in enhanced anomaly detection: {e}")
            return f"Error analyzing anomalies: {str(e)}"

    def _analyze_gps_anomalies(self, gps_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detailed GPS-specific anomaly analysis."""
        issues = {'critical': [], 'warning': [], 'info': []}
        
        try:
            # Check for GPS signal loss
            if 'Status' in gps_df.columns:
                signal_loss_points = gps_df[gps_df['Status'] < 3]  # Less than 3D fix
                if len(signal_loss_points) > 0:
                    first_loss = signal_loss_points.iloc[0]
                    if 'timestamp' in gps_df.columns:
                        loss_time = first_loss['timestamp'].strftime("%H:%M:%S")
                        issues['critical'].append(f"GPS signal lost at {loss_time} (Status: {first_loss['Status']})")
                    else:
                        issues['critical'].append(f"GPS signal degraded {len(signal_loss_points)} times")
            
            # Check satellite count
            if 'NSats' in gps_df.columns:
                low_sat_points = gps_df[gps_df['NSats'] < 6]
                if len(low_sat_points) > 0:
                    min_sats = gps_df['NSats'].min()
                    issues['warning'].append(f"Low satellite count: minimum {min_sats} satellites")
            
            # Check HDOP (horizontal dilution of precision)
            if 'HDop' in gps_df.columns:
                high_hdop = gps_df[gps_df['HDop'] > 2.0]
                if len(high_hdop) > 0:
                    max_hdop = gps_df['HDop'].max()
                    issues['warning'].append(f"Poor GPS precision: max HDOP {max_hdop:.1f}")
            
            # Check for altitude jumps
            if 'Alt' in gps_df.columns and len(gps_df) > 1:
                alt_diff = gps_df['Alt'].diff().abs()
                large_jumps = alt_diff[alt_diff > 50]  # 50m jumps
                if len(large_jumps) > 0:
                    max_jump = alt_diff.max()
                    issues['warning'].append(f"GPS altitude jumps detected: max {max_jump:.1f}m")
            
            # Check ground speed anomalies
            if 'Spd' in gps_df.columns:
                high_speed = gps_df[gps_df['Spd'] > 50]  # > 50 m/s seems excessive
                if len(high_speed) > 0:
                    max_speed = gps_df['Spd'].max()
                    issues['warning'].append(f"High speed readings: max {max_speed:.1f} m/s")
            
        except Exception as e:
            issues['warning'].append(f"GPS analysis error: {str(e)}")
        
        return issues

    def _analyze_attitude_anomalies(self, att_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze attitude control anomalies."""
        issues = {'critical': [], 'warning': [], 'info': []}
        
        try:
            # Check for excessive roll/pitch
            for angle in ['Roll', 'Pitch']:
                if angle in att_df.columns:
                    max_angle = att_df[angle].abs().max()
                    if max_angle > 45:  # Degrees
                        issues['critical'].append(f"Extreme {angle.lower()}: {max_angle:.1f}°")
                    elif max_angle > 30:
                        issues['warning'].append(f"High {angle.lower()}: {max_angle:.1f}°")
            
            # Check attitude error
            if 'ErrRP' in att_df.columns:
                high_error = att_df[att_df['ErrRP'].abs() > 20]
                if len(high_error) > 0:
                    max_error = att_df['ErrRP'].abs().max()
                    issues['warning'].append(f"Attitude control errors: max {max_error:.1f}°")
            
            # Check for oscillations
            if 'Roll' in att_df.columns and len(att_df) > 10:
                roll_std = att_df['Roll'].rolling(window=10).std().max()
                if roll_std > 10:
                    issues['warning'].append(f"Roll oscillations detected: std {roll_std:.1f}°")
                    
        except Exception as e:
            issues['warning'].append(f"Attitude analysis error: {str(e)}")
        
        return issues

    def _analyze_power_anomalies(self, curr_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze power system anomalies."""
        issues = {'critical': [], 'warning': [], 'info': []}
        
        try:
            # Check battery voltage
            if 'Volt' in curr_df.columns:
                min_volt = curr_df['Volt'].min()
                if min_volt < 10.5:  # Low voltage threshold
                    issues['critical'].append(f"Low battery voltage: {min_volt:.1f}V")
                elif min_volt < 11.1:
                    issues['warning'].append(f"Battery voltage getting low: {min_volt:.1f}V")
            
            # Check current spikes
            if 'Curr' in curr_df.columns:
                max_current = curr_df['Curr'].max()
                mean_current = curr_df['Curr'].mean()
                if max_current > mean_current * 3:
                    issues['warning'].append(f"Current spikes: max {max_current:.1f}A (avg {mean_current:.1f}A)")
                    
        except Exception as e:
            issues['warning'].append(f"Power analysis error: {str(e)}")
        
        return issues

    def _analyze_general_anomalies(self, df: pd.DataFrame, system_name: str) -> Dict[str, List[str]]:
        """General anomaly detection for other systems."""
        issues = {'warning': [], 'info': []}
        
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 10:  # Need sufficient data
                        mean_val = values.mean()
                        std_val = values.std()
                        outliers = values[abs(values - mean_val) > 3 * std_val]
                        
                        if len(outliers) > len(values) * 0.05:  # > 5% outliers
                            issues['warning'].append(f"{system_name}.{col}: {len(outliers)} anomalous readings")
                        elif len(outliers) > 0:
                            issues['info'].append(f"{system_name}.{col}: {len(outliers)} minor outliers")
                            
        except Exception as e:
            issues['warning'].append(f"{system_name} analysis error: {str(e)}")
        
        return issues

    def _compare_metrics(self, session: V2ConversationSession, metrics: List[str], comparison_type: str) -> str:
        """Compare different metrics in the flight data."""
        try:
            results = []
            
            for metric in metrics:
                if '.' in metric:
                    msg_type, col = metric.split('.', 1)
                    if msg_type in session.dataframes and col in session.dataframes[msg_type].columns:
                        values = session.dataframes[msg_type][col].dropna()
                        results.append({
                            'metric': metric,
                            'mean': values.mean(),
                            'max': values.max(),
                            'min': values.min(),
                            'count': len(values)
                        })
            
            if len(results) < 2:
                return "Need at least 2 valid metrics to compare."
            
            comparison_result = f"Comparison of {len(results)} metrics:\n"
            for result in results:
                comparison_result += f"- {result['metric']}: avg={result['mean']:.2f}, range={result['min']:.2f} to {result['max']:.2f}\n"
            
            return comparison_result
            
        except Exception as e:
            return f"Error comparing metrics: {str(e)}"

    def _generate_insights(self, session: V2ConversationSession, focus: str) -> str:
        """Generate comprehensive insights about the flight."""
        try:
            insights = []
            
            # Flight duration
            if any('timestamp' in df.columns for df in session.dataframes.values()):
                timestamps = []
                for df in session.dataframes.values():
                    if 'timestamp' in df.columns:
                        timestamps.extend(df['timestamp'].dropna().tolist())
                
                if timestamps:
                    duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
                    insights.append(f"Flight duration: {duration:.1f} minutes")
            
            # Altitude insights
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Alt' in gps_df.columns:
                    max_alt = gps_df['Alt'].max()
                    avg_alt = gps_df['Alt'].mean()
                    insights.append(f"Maximum altitude: {max_alt:.1f}m, Average altitude: {avg_alt:.1f}m")
            
            # Speed insights
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Spd' in gps_df.columns:
                    max_speed = gps_df['Spd'].max()
                    avg_speed = gps_df['Spd'].mean()
                    insights.append(f"Maximum speed: {max_speed:.1f} m/s, Average speed: {avg_speed:.1f} m/s")
            
            # Mode changes
            if 'MODE' in session.dataframes:
                mode_df = session.dataframes['MODE']
                mode_changes = len(mode_df)
                insights.append(f"Flight mode changes: {mode_changes}")
            
            if not insights:
                insights.append("Basic flight data processed successfully")
            
            return "Flight Analysis Summary:\n" + "\n".join(f"• {insight}" for insight in insights)
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def _detect_flight_events(self, session: V2ConversationSession, event_types: List[str]) -> str:
        """Detect specific flight events like GPS loss, mode changes, critical alerts with timestamps."""
        events = []
        
        try:
            for event_type in event_types:
                if event_type == 'gps_loss' and 'GPS' in session.dataframes:
                    gps_events = self._detect_gps_loss_events(session.dataframes['GPS'])
                    events.extend(gps_events)
                
                elif event_type == 'mode_changes' and 'MODE' in session.dataframes:
                    mode_events = self._detect_mode_changes(session.dataframes['MODE'])
                    events.extend(mode_events)
                
                elif event_type == 'critical_alerts' and 'MSG' in session.dataframes:
                    alert_events = self._detect_critical_alerts(session.dataframes['MSG'])
                    events.extend(alert_events)
                
                elif event_type == 'power_issues' and 'CURR' in session.dataframes:
                    power_events = self._detect_power_issues(session.dataframes['CURR'])
                    events.extend(power_events)
                
                elif event_type == 'attitude_problems' and 'ATT' in session.dataframes:
                    attitude_events = self._detect_attitude_problems(session.dataframes['ATT'])
                    events.extend(attitude_events)
            
            if not events:
                return "No significant flight events detected."
            
            # Sort events by timestamp
            events.sort(key=lambda x: x.get('timestamp', 0))
            
            # Format response
            response = "FLIGHT EVENTS DETECTED:\n"
            for event in events:
                if 'time_str' in event:
                    response += f"{event['time_str']}: {event['description']}\n"
                else:
                    response += f"• {event['description']}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error detecting flight events: {e}")
            return f"Error detecting flight events: {str(e)}"

    def _detect_gps_loss_events(self, gps_df: pd.DataFrame) -> List[Dict]:
        """Detect GPS signal loss events with timestamps."""
        events = []
        
        try:
            if 'Status' in gps_df.columns:
                # Find transitions to poor GPS status
                prev_status = None
                for idx, row in gps_df.iterrows():
                    current_status = row['Status']
                    
                    # GPS signal lost (status < 3 means no 3D fix)
                    if prev_status is not None and prev_status >= 3 and current_status < 3:
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in gps_df.columns else "Unknown time"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"GPS signal lost (Status: {current_status})",
                            'severity': 'critical'
                        })
                    
                    # GPS signal recovered
                    elif prev_status is not None and prev_status < 3 and current_status >= 3:
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in gps_df.columns else "Unknown time"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"GPS signal recovered (Status: {current_status})",
                            'severity': 'info'
                        })
                    
                    prev_status = current_status
        except Exception as e:
            logger.error(f"Error detecting GPS events: {e}")
        
        return events

    def _detect_mode_changes(self, mode_df: pd.DataFrame) -> List[Dict]:
        """Detect flight mode changes."""
        events = []
        
        try:
            mode_names = {
                0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
                5: "LOITER", 6: "RTL", 7: "CIRCLE", 8: "POSITION", 9: "LAND",
                10: "AUTOTUNE", 11: "POSHOLD", 12: "BRAKE", 13: "THROW",
                14: "AVOID_ADSB", 15: "GUIDED_NOGPS", 16: "SMART_RTL",
                17: "FLOWHOLD", 18: "FOLLOW", 19: "ZIGZAG", 20: "SYSTEMID",
                21: "AUTOROTATE", 22: "AUTO_RTL"
            }
            
            for idx, row in mode_df.iterrows():
                mode_num = row.get('Mode', 0)
                mode_name = mode_names.get(mode_num, f"MODE_{mode_num}")
                time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in mode_df.columns else "Unknown time"
                
                events.append({
                    'timestamp': row.get('timestamp', 0),
                    'time_str': time_str,
                    'description': f"Mode changed to {mode_name}",
                    'severity': 'info'
                })
        except Exception as e:
            logger.error(f"Error detecting mode changes: {e}")
        
        return events

    def _detect_critical_alerts(self, msg_df: pd.DataFrame) -> List[Dict]:
        """Detect critical alert messages."""
        events = []
        
        try:
            critical_keywords = ['ERROR', 'CRITICAL', 'FAIL', 'EMERGENCY', 'LOST', 'WARNING']
            
            for idx, row in msg_df.iterrows():
                if 'Message' in row:
                    message = str(row['Message']).upper()
                    if any(keyword in message for keyword in critical_keywords):
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in msg_df.columns else "Unknown time"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f" Alert: {row['Message']}",
                            'severity': 'warning'
                        })
        except Exception as e:
            logger.error(f"Error detecting critical alerts: {e}")
        
        return events

    def _detect_power_issues(self, curr_df: pd.DataFrame) -> List[Dict]:
        """Detect power-related issues."""
        events = []
        
        try:
            if 'Volt' in curr_df.columns:
                low_voltage_threshold = 10.5
                critical_voltage_threshold = 10.0
                
                for idx, row in curr_df.iterrows():
                    voltage = row['Volt']
                    time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in curr_df.columns else "Unknown time"
                    
                    if voltage < critical_voltage_threshold:
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Critical low voltage: {voltage:.1f}V",
                            'severity': 'critical'
                        })
                    elif voltage < low_voltage_threshold:
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Low voltage warning: {voltage:.1f}V",
                            'severity': 'warning'
                        })
        except Exception as e:
            logger.error(f"Error detecting power issues: {e}")
        
        return events

    def _detect_attitude_problems(self, att_df: pd.DataFrame) -> List[Dict]:
        """Detect attitude control problems."""
        events = []
        
        try:
            for idx, row in att_df.iterrows():
                time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in att_df.columns else "Unknown time"
                
                # Check for extreme angles
                if 'Roll' in row and abs(row['Roll']) > 45:
                    events.append({
                        'timestamp': row.get('timestamp', 0),
                        'time_str': time_str,
                        'description': f"Extreme roll angle: {row['Roll']:.1f}°",
                        'severity': 'critical'
                    })
                
                if 'Pitch' in row and abs(row['Pitch']) > 45:
                    events.append({
                        'timestamp': row.get('timestamp', 0),
                        'time_str': time_str,
                        'description': f"Extreme pitch angle: {row['Pitch']:.1f}°",
                        'severity': 'critical'
                    })
        except Exception as e:
            logger.error(f"Error detecting attitude problems: {e}")
        
        return events

    def _analyze_flight_phase(self, session: V2ConversationSession, phase: str, metrics: List[str]) -> str:
        """Analyze specific phases of flight (takeoff, cruise, landing) with detailed metrics."""
        try:
            if phase == 'all':
                # Analyze all phases
                result = "🛫 COMPLETE FLIGHT ANALYSIS:\n\n"
                for flight_phase in ['takeoff', 'cruise', 'landing']:
                    phase_analysis = self._analyze_single_phase(session, flight_phase, metrics)
                    result += f"{'='*40}\n{phase_analysis}\n"
                return result
            else:
                return self._analyze_single_phase(session, phase, metrics)
                
        except Exception as e:
            logger.error(f"Error analyzing flight phase: {e}")
            return f"Error analyzing flight phase: {str(e)}"

    def _analyze_single_phase(self, session: V2ConversationSession, phase: str, metrics: List[str]) -> str:
        """Analyze a single flight phase."""
        try:
            # Get phase time ranges based on altitude and mode data
            phase_data = self._identify_flight_phase(session, phase)
            
            if not phase_data:
                return f"Could not identify {phase} phase in flight data."
            
            result = f"{phase.upper()} PHASE ANALYSIS:\n"
            result += f"Duration: {phase_data['duration']:.1f} minutes\n"
            result += f"Time range: {phase_data['start_time']} - {phase_data['end_time']}\n\n"
            
            # Analyze requested metrics
            for metric in metrics:
                metric_analysis = self._analyze_phase_metric(session, phase_data, metric)
                result += f"{metric_analysis}\n"
            
            return result
            
        except Exception as e:
            return f"Error analyzing {phase} phase: {str(e)}"

    def _identify_flight_phase(self, session: V2ConversationSession, phase: str) -> Dict[str, Any]:
        """Identify time ranges for different flight phases."""
        try:
            if 'GPS' not in session.dataframes:
                return None
                
            gps_df = session.dataframes['GPS']
            if 'Alt' not in gps_df.columns or 'timestamp' not in gps_df.columns:
                return None
            
            # Basic phase detection based on altitude changes
            altitudes = gps_df['Alt'].dropna()
            timestamps = gps_df['timestamp'].dropna()
            
            if len(altitudes) < 10:
                return None
            
            min_alt = altitudes.min()
            max_alt = altitudes.max()
            alt_range = max_alt - min_alt
            
            if phase == 'takeoff':
                # First 20% of altitude gain
                takeoff_alt = min_alt + (alt_range * 0.2)
                takeoff_indices = altitudes[altitudes <= takeoff_alt].index
                if len(takeoff_indices) > 0:
                    start_idx = takeoff_indices[0]
                    end_idx = takeoff_indices[-1]
                    start_time = timestamps.iloc[start_idx]
                    end_time = timestamps.iloc[end_idx]
                    duration = (end_time - start_time).total_seconds() / 60
                    
                    return {
                        'start_time': start_time.strftime("%H:%M:%S"),
                        'end_time': end_time.strftime("%H:%M:%S"),
                        'duration': duration,
                        'data_range': (start_idx, end_idx)
                    }
            
            elif phase == 'cruise':
                # Middle 60% of flight time
                total_indices = len(timestamps)
                start_idx = int(total_indices * 0.2)
                end_idx = int(total_indices * 0.8)
                start_time = timestamps.iloc[start_idx]
                end_time = timestamps.iloc[end_idx]
                duration = (end_time - start_time).total_seconds() / 60
                
                return {
                    'start_time': start_time.strftime("%H:%M:%S"),
                    'end_time': end_time.strftime("%H:%M:%S"),
                    'duration': duration,
                    'data_range': (start_idx, end_idx)
                }
            
            elif phase == 'landing':
                # Last 20% of altitude loss
                landing_alt = min_alt + (alt_range * 0.2)
                landing_indices = altitudes[altitudes <= landing_alt].index
                if len(landing_indices) > 0:
                    start_idx = landing_indices[0]
                    end_idx = landing_indices[-1]
                    start_time = timestamps.iloc[start_idx]
                    end_time = timestamps.iloc[end_idx]
                    duration = (end_time - start_time).total_seconds() / 60
                    
                    return {
                        'start_time': start_time.strftime("%H:%M:%S"),
                        'end_time': end_time.strftime("%H:%M:%S"),
                        'duration': duration,
                        'data_range': (start_idx, end_idx)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying flight phase: {e}")
            return None

    def _analyze_phase_metric(self, session: V2ConversationSession, phase_data: Dict, metric: str) -> str:
        """Analyze a specific metric during a flight phase."""
        try:
            start_idx, end_idx = phase_data['data_range']
            
            if metric == 'altitude' and 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Alt' in gps_df.columns:
                    phase_alt = gps_df['Alt'].iloc[start_idx:end_idx]
                    return f"Altitude: {phase_alt.min():.1f}m - {phase_alt.max():.1f}m (avg: {phase_alt.mean():.1f}m)"
            
            elif metric == 'speed' and 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Spd' in gps_df.columns:
                    phase_speed = gps_df['Spd'].iloc[start_idx:end_idx]
                    return f"Speed: max {phase_speed.max():.1f} m/s, avg {phase_speed.mean():.1f} m/s"
            
            elif metric == 'power' and 'CURR' in session.dataframes:
                curr_df = session.dataframes['CURR']
                phase_curr = curr_df.iloc[start_idx:end_idx] if len(curr_df) > end_idx else curr_df
                if 'Volt' in phase_curr.columns:
                    voltage = phase_curr['Volt']
                    return f"Power: {voltage.min():.1f}V - {voltage.max():.1f}V (avg: {voltage.mean():.1f}V)"
            
            elif metric == 'stability' and 'ATT' in session.dataframes:
                att_df = session.dataframes['ATT']
                phase_att = att_df.iloc[start_idx:end_idx] if len(att_df) > end_idx else att_df
                if 'Roll' in phase_att.columns and 'Pitch' in phase_att.columns:
                    roll_std = phase_att['Roll'].std()
                    pitch_std = phase_att['Pitch'].std()
                    return f"Stability: Roll std {roll_std:.1f}°, Pitch std {pitch_std:.1f}°"
            
            return f"{metric}: Data not available for this phase"
            
        except Exception as e:
            return f"{metric}: Analysis error - {str(e)}"

    def _get_timeline_analysis(self, session: V2ConversationSession, time_resolution: str) -> str:
        """Provide a chronological timeline of key events and issues during the flight."""
        try:
            # Collect all events from different sources
            all_events = []
            
            # GPS events
            if 'GPS' in session.dataframes:
                gps_events = self._get_gps_timeline_events(session.dataframes['GPS'])
                all_events.extend(gps_events)
            
            # Mode changes
            if 'MODE' in session.dataframes:
                mode_events = self._get_mode_timeline_events(session.dataframes['MODE'])
                all_events.extend(mode_events)
            
            # Power events
            if 'CURR' in session.dataframes:
                power_events = self._get_power_timeline_events(session.dataframes['CURR'])
                all_events.extend(power_events)
            
            # Critical messages
            if 'MSG' in session.dataframes:
                msg_events = self._get_message_timeline_events(session.dataframes['MSG'])
                all_events.extend(msg_events)
            
            if not all_events:
                return "No significant timeline events found."
            
            # Sort by timestamp
            all_events.sort(key=lambda x: x.get('timestamp', 0))
            
            # Group by time resolution
            if time_resolution == 'auto':
                time_resolution = 'minutes' if len(all_events) > 50 else 'seconds'
            
            # Format timeline
            result = f"FLIGHT TIMELINE ({time_resolution} resolution):\n\n"
            
            prev_time_group = None
            for event in all_events:
                if 'time_str' in event:
                    if time_resolution == 'minutes':
                        time_group = event['time_str'][:5]  # HH:MM
                    else:
                        time_group = event['time_str']  # HH:MM:SS
                    
                    if time_group != prev_time_group:
                        result += f"\n{time_group}:\n"
                        prev_time_group = time_group
                    
                    result += f"   {event['description']}\n"
                else:
                    result += f"• {event['description']}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating timeline: {e}")
            return f"Error generating timeline: {str(e)}"

    def _get_gps_timeline_events(self, gps_df: pd.DataFrame) -> List[Dict]:
        """Get GPS-related timeline events."""
        events = []
        
        try:
            # Track GPS status changes
            if 'Status' in gps_df.columns:
                prev_status = None
                for idx, row in gps_df.iterrows():
                    current_status = row['Status']
                    if prev_status is not None and current_status != prev_status:
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in gps_df.columns else "Unknown"
                        if current_status < 3:
                            events.append({
                                'timestamp': row.get('timestamp', 0),
                                'time_str': time_str,
                                'description': f"GPS signal degraded (Status: {current_status})",
                                'type': 'gps'
                            })
                        elif prev_status < 3:
                            events.append({
                                'timestamp': row.get('timestamp', 0),
                                'time_str': time_str,
                                'description': f"GPS signal recovered (Status: {current_status})",
                                'type': 'gps'
                            })
                    prev_status = current_status
        except Exception as e:
            logger.error(f"Error getting GPS timeline: {e}")
        
        return events

    def _get_mode_timeline_events(self, mode_df: pd.DataFrame) -> List[Dict]:
        """Get mode change timeline events."""
        events = []
        
        try:
            mode_names = {
                0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
                5: "LOITER", 6: "RTL", 7: "CIRCLE", 8: "POSITION", 9: "LAND"
            }
            
            for idx, row in mode_df.iterrows():
                mode_num = row.get('Mode', 0)
                mode_name = mode_names.get(mode_num, f"MODE_{mode_num}")
                time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in mode_df.columns else "Unknown"
                
                events.append({
                    'timestamp': row.get('timestamp', 0),
                    'time_str': time_str,
                    'description': f"Flight mode: {mode_name}",
                    'type': 'mode'
                })
        except Exception as e:
            logger.error(f"Error getting mode timeline: {e}")
        
        return events

    def _get_power_timeline_events(self, curr_df: pd.DataFrame) -> List[Dict]:
        """Get power-related timeline events."""
        events = []
        
        try:
            if 'Volt' in curr_df.columns:
                # Sample power events (e.g., every 100 readings or significant changes)
                sample_interval = max(1, len(curr_df) // 20)  # Max 20 power readings
                for i in range(0, len(curr_df), sample_interval):
                    row = curr_df.iloc[i]
                    voltage = row['Volt']
                    time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in curr_df.columns else "Unknown"
                    
                    if voltage < 11.0:  # Only report lower voltages
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Battery voltage: {voltage:.1f}V",
                            'type': 'power'
                        })
        except Exception as e:
            logger.error(f"Error getting power timeline: {e}")
        
        return events

    def _get_message_timeline_events(self, msg_df: pd.DataFrame) -> List[Dict]:
        """Get message timeline events."""  
        events = []
        
        try:
            for idx, row in msg_df.iterrows():
                if 'Message' in row:
                    time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in msg_df.columns else "Unknown"
                    events.append({
                        'timestamp': row.get('timestamp', 0),
                        'time_str': time_str,
                        'description': f"System message: {row['Message']}",
                        'type': 'message'
                    })
        except Exception as e:
            logger.error(f"Error getting message timeline: {e}")
        
        return events

# Instantiate a singleton of the service for the main app to import
chat_service_v2 = ChatServiceV2() 