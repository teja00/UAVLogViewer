"""
AI Chat Service V2 with Agentic Tooling for UAV Log Analysis

This service introduces an agentic approach to telemetry data analysis.
Instead of just summarizing data, it dynamically generates and executes
Python code to answer user queries, leveraging pandas DataFrames for powerful,
in-memory data manipulation. This allows for far more specific and complex
data analysis questions to be answered.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import json
import pandas as pd
from io import StringIO
import traceback

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


# A more specific session state for V2 is now in models.py


class ChatServiceV2:
    """
    An agentic chat service that uses OpenAI's tool-calling feature to
    generate and execute Python code for telemetry data analysis.
    """

    def __init__(self):
        """
        Initializes the service, loading settings and setting up the OpenAI client.
        """
        self.settings = get_settings()
        # Using the synchronous client here for simplicity in the exec environment.
        # A fully async implementation would require more complex handling of exec.
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured. V2 service will not be functional.")
        self.sessions: Dict[str, V2ConversationSession] = {}
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

    def _get_system_prompt(self, session: V2ConversationSession) -> str:
        """
        Generates a detailed system prompt for the AI, instructing it on how
        to act as a data analysis agent.

        This prompt is critical. It defines the AI's role, its available
        tools (the Python code executor), and the data context (the schemas
        of the pandas DataFrames).
        """
        if not session.dataframes:
            return "You are a helpful assistant." # Fallback prompt

        prompt = """You are an expert UAV flight data analysis agent. Your goal is to answer user questions by writing and executing Python code.

You have access to a set of pandas DataFrames, loaded from the user's flight log.
The available DataFrames and their schemas are as follows:
"""

        for name, schema in session.dataframe_schemas.items():
            prompt += f"\n- DataFrame `dfs['{name}']`:\n"
            prompt += f"  - Columns and types: {schema['columns']}\n"
            if 'description' in schema:
                prompt += f"  - Description: {schema['description']}\n"
        
        prompt += """

For context, here is a summary of what the different ArduPilot message types (which correspond to the DataFrames) mean, based on the official ArduPilot documentation.
This can help you understand the columns and data in each DataFrame.
"""

        for msg_type, description in self.ardupilot_messages.items():
            if msg_type in session.dataframes:
                 prompt += f"\n- {msg_type}: {description}"

        prompt += """

To answer the user's question, you MUST use the `execute_python_code` tool.
The tool will execute the Python code you provide and return the result.

Your Python code can perform calculations, data filtering, and analysis on the DataFrames.
The DataFrames are accessible via a dictionary called `dfs`. For example, to access the GPS data, use `dfs['GPS']`.

IMPORTANT RULES:
1.  Your code MUST NOT make any external API calls, file system changes, or use any libraries other than pandas, numpy, and standard Python libraries.
2.  The LAST line of your code must be an expression that evaluates to the result you want to return. This result will be captured and shown to the user.
3.  Do not just return a DataFrame. Return a specific, calculated value (e.g., a number, a string, a list, or a dictionary of results). For example, instead of returning the whole GPS dataframe, calculate the max altitude using `dfs['GPS']['alt'].max()`
4.  If asked to plot, you cannot generate an image. Instead, describe the plot's findings in text or return a dictionary with the data needed to create the plot.
5.  Always assume the user's question is about the provided flight data.
"""
        return prompt

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
        Main chat method for the agentic service.
        This method now checks the processing state before attempting to answer.
        """
        logger.info(f"[{session_id}] Chat request received: '{user_message[:50]}...'")
        
        if not self.client:
            logger.error("OpenAI client is not configured")
            return "OpenAI client is not configured. This service is unavailable."

        session = await self.create_or_get_session(session_id)
        
        logger.info(f"[{session_id}] Session status - Processing: {session.is_processing}, DataFrames: {len(session.dataframes)}, Error: {session.processing_error}")

        if session.is_processing:
            logger.info(f"[{session_id}] Session still processing, returning wait message")
            return "Your log file is still being analyzed. Please wait a moment and try again."

        if session.processing_error:
            logger.error(f"[{session_id}] Processing error found: {session.processing_error}")
            return f"Sorry, an error occurred while processing your log file: {session.processing_error}"

        if not session.dataframes:
            logger.warning(f"[{session_id}] No dataframes found in session")
            return "No telemetry data is loaded in the session. Please upload a log file first."

        session.messages.append(ChatMessage(role="user", content=user_message))

        try:
            # Step 1: Call the model with the user query and tools
            response = self.client.chat.completions.create(
                model=self.settings.openai_model, # Use model from settings
                messages=[
                    {"role": "system", "content": self._get_system_prompt(session)},
                    {"role": "user", "content": user_message}
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "execute_python_code",
                        "description": "Executes Python code to analyze the flight data DataFrames.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "The Python code to execute. Must return a value.",
                                }
                            },
                            "required": ["code"],
                        },
                    },
                }],
                tool_choice={"type": "function", "function": {"name": "execute_python_code"}},
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Step 2: Check if the model wants to use a tool
            if not tool_calls:
                return "The AI agent chose not to use the analysis tool. Please rephrase your question to be more specific about the data."

            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            
            if function_name != "execute_python_code":
                return f"Error: The AI agent requested an unknown tool: {function_name}"

            # Step 3: Execute the code from the tool call
            function_args = json.loads(tool_call.function.arguments)
            code_to_execute = function_args["code"]
            
            # This is where the magic happens: executing the AI-generated code.
            # WARNING: `exec` is powerful but can be dangerous if the code is not
            # properly sandboxed. For a production system, this execution
            # environment should be heavily restricted (e.g., run in a separate
            # process with no file system or network access).
            logger.info(f"Executing AI-generated code for session {session_id}:\n{code_to_execute}")
            
            # We use `exec` to run the code and `eval` on the last line to get the result
            try:
                # The `exec` environment
                local_scope = {"dfs": session.dataframes, "pd": pd}
                
                # Execute the code block. The last expression is evaluated as the result.
                # Note: This is a simplified execution model.
                code_lines = code_to_execute.strip().split('\n')
                
                # If there are multiple lines, exec all but the last
                if len(code_lines) > 1:
                    exec('\n'.join(code_lines[:-1]), {}, local_scope)
                
                # Evaluate the last line to get the result
                result = eval(code_lines[-1], {}, local_scope)

                tool_output = f"Execution successful.\nResult: {str(result)}"

            except Exception as e:
                logger.error(f"Error executing AI-generated code: {e}")
                tool_output = f"Execution failed.\nError: {traceback.format_exc()}"

            # Step 4: Send the result back to the model to get a natural language response
            final_response = self.client.chat.completions.create(
                model=self.settings.openai_model, # Use model from settings
                messages=[
                    {"role": "system", "content": self._get_system_prompt(session)},
                    {"role": "user", "content": user_message},
                    response_message, # Include the model's previous message
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_output,
                    },
                ],
            )
            
            final_answer = final_response.choices[0].message.content
            session.messages.append(ChatMessage(role="assistant", content=final_answer))
            return final_answer

        except Exception as e:
            logger.error(f"Error in V2 chat processing for session {session_id}: {e}")
            return "An unexpected error occurred while processing your request with the V2 agent."

# Instantiate a singleton of the service for the main app to import
chat_service_v2 = ChatServiceV2() 