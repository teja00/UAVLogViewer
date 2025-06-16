"""
Multi-role LLM agent implementation for UAV log analysis using CrewAI.
This provides a sophisticated agent architecture with separate roles for planning, execution, and critique.
Each role can iterate until it achieves its goal with dynamic stopping conditions.

Usage:
    # Replace the existing ChatAgentV2 with MultiRoleAgent
    from backend.agent.multi_role_agent import MultiRoleAgent
    
    agent = MultiRoleAgent()
    result = await agent.chat(session_id, user_message)
"""

import logging
import uuid
import json
import asyncio
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import re
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import pandas as pd

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from config import get_settings
from models import ChatMessage, V2ConversationSession

from utils.documentation import DocumentationService
from utils.log_parser import LogParserService, get_data_summary
from tools.tool_definitions import get_tool_definitions  
from tools.analysis_tools import AnalysisTools

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enumeration of agent roles."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"


class IterationStatus(Enum):
    """Status of iterative execution."""
    CONTINUE = "continue"
    COMPLETE = "complete"
    FAILED = "failed"
    NEEDS_REPLAN = "needs_replan"


@dataclass
class LLMCall:
    """Track individual LLM calls for cost and performance monitoring."""
    role: str
    model: str
    tokens_prompt: int
    tokens_completion: int
    cost_estimate: float
    duration_ms: int
    timestamp: datetime
    session_id: str
    iteration: int = 0


@dataclass
class IterationResult:
    """Result from a single iteration of any role."""
    status: IterationStatus
    content: str
    reasoning: str
    confidence: float
    next_action: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Structured plan from the Planner role."""
    needs_tools: bool
    tool_sequence: List[str]
    reasoning: str
    confidence: float
    approach: str
    estimated_complexity: str  # low, medium, high
    requires_critique: bool = True
    max_iterations: int = 5  # Dynamic iteration limit
    success_criteria: List[str] = field(default_factory=list)  # What constitutes success
    # Enhanced dataframe selection
    target_dataframes: List[str] = field(default_factory=list)
    target_columns: Dict[str, List[str]] = field(default_factory=dict)
    data_relationships: List[str] = field(default_factory=list)
    # Multi-step planning
    execution_steps: List[str] = field(default_factory=list)  # Ordered steps to execute
    stopping_conditions: List[str] = field(default_factory=list)  # When to stop


@dataclass
class ExecutionResult:
    """Result from the Executor role."""
    answer: str
    tools_used: List[str]
    execution_success: bool
    iterations: int = 0
    final_confidence: float = 0.0
    success_criteria_met: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class CritiqueResult:
    """Result from the Critic role."""
    final_answer: str
    changes_made: bool
    critique_summary: str
    quality_score: float  # 0.0 - 1.0
    iterations: int = 0
    needs_re_execution: bool = False
    re_execution_instructions: Optional[str] = None


from pydantic import BaseModel, Field
from typing import Type

# Input schemas for tools
class PythonCodeInput(BaseModel):
    """Input schema for Python code execution."""
    code: str = Field(..., description="Python code to execute for data analysis")

class AltitudeQueryInput(BaseModel):
    """Input schema for altitude queries."""
    query: str = Field(..., description="Altitude-related query like 'maximum altitude' or 'highest point'")

class TelemetryQueryInput(BaseModel):
    """Input schema for telemetry analysis."""
    query: str = Field(..., description="Query about flight data analysis")

# Custom CrewAI Tools with simplified implementation
class PythonCodeExecutorTool(BaseTool):
    name: str = "execute_python_code"
    description: str = "Execute Python code to analyze UAV flight data and extract specific numerical values, statistics, and insights. Pass a natural language query like 'What was the maximum battery temperature?' rather than incomplete Python code."
    args_schema: Type[BaseModel] = PythonCodeInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, code: str) -> str:
        """Execute Python code or auto-generate code based on common queries."""
        logger.info(f"[{self._session.session_id}] PythonCodeExecutorTool received: {code[:100]}...")
        
        # Debug: Check if session has dataframes
        if not self._session.dataframes:
            logger.error(f"[{self._session.session_id}] NO DATAFRAMES AVAILABLE - session.dataframes is empty!")
            return "Error: No flight data available. Please upload a log file first."
        
        logger.info(f"[{self._session.session_id}] Available dataframes: {list(self._session.dataframes.keys())}")
        
        try:
            result = None
            
            # Enhanced query detection - check for natural language first
            if self._is_query_not_code(code):
                logger.info(f"[{self._session.session_id}] Detected natural language query, generating code...")
                # Auto-generate Python code for common queries
                generated_code = self._generate_code_for_query(code)
                if generated_code:
                    logger.info(f"[{self._session.session_id}] Generated Python code: {generated_code[:200]}...")
                    result = self._analysis_tools.execute_python_code(self._session, generated_code)
                    logger.info(f"[{self._session.session_id}] Analysis tools returned: {result[:200]}...")
                else:
                    # Fallback: try to provide a direct answer based on query
                    result = self._attempt_direct_query_answer(code)
            
            # Check if this looks like incomplete/invalid Python code
            elif self._is_incomplete_code(code):
                logger.info(f"[{self._session.session_id}] Detected incomplete code, trying to fix or interpret...")
                # Try to extract the intent from incomplete code
                fixed_query = self._extract_intent_from_code(code)
                if fixed_query:
                    generated_code = self._generate_code_for_query(fixed_query)
                    if generated_code:
                        logger.info(f"[{self._session.session_id}] Fixed incomplete code and generated proper code")
                        result = self._analysis_tools.execute_python_code(self._session, generated_code)
                    else:
                        # Try direct analysis as fallback
                        result = self._attempt_direct_query_answer(fixed_query)
                else:
                    # Enhanced fallback: try to interpret the variable assignment as a query
                    result = self._interpret_assignment_as_query(code)
            
            else:
                # Execute the provided Python code directly
                logger.info(f"[{self._session.session_id}] Executing provided Python code directly")
                result = self._analysis_tools.execute_python_code(self._session, code)
            
            # Validate and return result
            if result and len(result.strip()) > 0:
                logger.info(f"[{self._session.session_id}] Returning successful result: {result[:200]}...")
                return result
            else:
                logger.warning(f"[{self._session.session_id}] Empty or null result, trying fallback...")
                return self._attempt_direct_query_answer(code)
            
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            logger.error(f"[{self._session.session_id}] {error_msg}")
            # Try one more fallback
            try:
                fallback_result = self._attempt_direct_query_answer(code)
                logger.info(f"[{self._session.session_id}] Fallback successful: {fallback_result[:100]}...")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"[{self._session.session_id}] Fallback also failed: {str(fallback_error)}")
                return f"Analysis failed: {error_msg}. Available data: {list(self._session.dataframes.keys())}"
    
    def _attempt_direct_query_answer(self, query: str) -> str:
        """Attempt to provide a direct answer based on available data when code generation fails."""
        try:
            query_lower = query.lower()
            available_dfs = list(self._session.dataframes.keys())
            
            # Quick altitude check
            if 'altitude' in query_lower or 'highest' in query_lower:
                for df_name in ['GPS', 'BARO', 'CTUN', 'AHR2']:
                    if df_name in available_dfs:
                        df = self._session.dataframes[df_name]
                        if 'Alt' in df.columns:
                            max_alt = df['Alt'].max()
                            return f"The highest altitude reached was {max_alt:.1f} meters (from {df_name} data)."
            
            # Quick error check  
            if 'error' in query_lower or 'warning' in query_lower:
                error_count = 0
                if 'ERR' in available_dfs:
                    error_count += len(self._session.dataframes['ERR'])
                if 'MSG' in available_dfs:
                    msg_df = self._session.dataframes['MSG']
                    if 'Message' in msg_df.columns:
                        error_msgs = msg_df[msg_df['Message'].str.contains('ERROR|WARNING|CRITICAL', case=False, na=False)]
                        error_count += len(error_msgs)
                
                if error_count > 0:
                    return f"Found {error_count} errors/warnings in the flight data."
                else:
                    return "No significant errors or warnings detected in the flight data."
            
            # If we can't provide a specific answer, at least show what data is available
            return f"Unable to analyze the query directly. Available data types in this flight log: {', '.join(available_dfs)}. Please provide a more specific question."
            
        except Exception as e:
            return f"Unable to process query: {str(e)}"
    
    def _is_query_not_code(self, text: str) -> bool:
        """Determine if text is a query rather than Python code."""
        # Check for common Python keywords
        python_keywords = ['def ', 'import ', 'for ', 'if ', 'while ', 'class ', 'print(', 'return', 'exec', 'eval']
        
        # If it contains Python keywords, treat as code
        if any(keyword in text for keyword in python_keywords):
            return False
        
        # If it's a question or natural language, treat as query
        question_indicators = ['what', 'when', 'how', 'where', 'maximum', 'minimum', 'battery', 'temperature', 'time', 'error', 'signal', 'loss', 'highest', 'lowest']
        
        return any(indicator in text.lower() for indicator in question_indicators)
    
    def _is_incomplete_code(self, text: str) -> bool:
        """Check if the code looks incomplete or invalid."""
        # Simple assignment without return/output
        if '=' in text and not any(keyword in text for keyword in ['return', 'print', 'f"', '"""', "'''"]):
            return True
        
        # Single variable name
        if len(text.strip().split()) == 1 and text.strip().isidentifier():
            return True
        
        # Try to compile/parse the code
        try:
            compile(text, '<string>', 'exec')
            # If it compiles but doesn't seem to return anything useful
            if not any(keyword in text for keyword in ['return', 'print', 'f"', '"""', "'''", 'result', 'output']):
                return True
        except:
            return True
        
        return False
    
    def _extract_intent_from_code(self, code: str) -> Optional[str]:
        """Try to extract the intent from incomplete code."""
        code_lower = code.lower()
        
        # Battery temperature
        if 'battery' in code_lower and 'temp' in code_lower:
            return "What was the maximum battery temperature?"
        
        # Flight time
        if 'flight' in code_lower and 'time' in code_lower:
            return "What was the total flight time?"
        
        # GPS signal
        if 'gps' in code_lower and ('signal' in code_lower or 'loss' in code_lower):
            return "Was there any GPS signal loss?"
        
        # Altitude
        if 'altitude' in code_lower or 'alt' in code_lower:
            if 'max' in code_lower or 'maximum' in code_lower:
                return "What was the maximum altitude?"
            elif 'min' in code_lower or 'minimum' in code_lower:
                return "What was the minimum altitude?"
        
        # Errors
        if 'error' in code_lower:
            return "Were there any critical errors?"
        
        return None
    
    def _generate_code_for_query(self, query: str) -> Optional[str]:
        """Generate Python code for common queries."""
        query_lower = query.lower()
        
        # Enhanced contextual altitude queries
        if any(phrase in query_lower for phrase in ['consistently high', 'high altitude', 'altitude consistency', 'stable altitude']):
            return self._generate_altitude_consistency_code()
        
        # Flight stability and control queries
        elif any(phrase in query_lower for phrase in ['flight stability', 'stable flight', 'flight control']):
            return self._generate_flight_stability_code()
        
        # Power consumption queries - NEW
        elif any(phrase in query_lower for phrase in ['power consumption', 'analyze power', 'battery consumption', 'current consumption', 'power analysis']):
            return self._generate_power_consumption_code()
        
        # Battery queries (temperature, voltage, current)
        elif 'battery' in query_lower:
            if 'temperature' in query_lower or 'temp' in query_lower:
                return self._generate_battery_temperature_code()
            elif 'voltage' in query_lower or 'volt' in query_lower:
                return self._generate_battery_voltage_code()
            elif 'current' in query_lower or 'consumption' in query_lower:
                return self._generate_power_consumption_code()
            else:
                return self._generate_comprehensive_battery_code()
        
        # Power system queries
        elif any(word in query_lower for word in ['power', 'voltage', 'current']) and not 'altitude' in query_lower:
            return self._generate_power_consumption_code()
        
        # Flight time queries - enhanced detection
        elif any(phrase in query_lower for phrase in ['flight time', 'total time', 'duration', 'how long', 'time was', 'minutes']):
            return self._generate_flight_time_code()
        
        # Flight summary queries that should include duration
        elif any(phrase in query_lower for phrase in ['summary', 'overview', 'give me a summary']) and 'flight' in query_lower:
            return self._generate_comprehensive_flight_summary_code()
        
        # Temperature queries - enhanced detection for maximum battery temperature
        elif any(phrase in query_lower for phrase in ['maximum battery temperature', 'max battery temp', 'battery temperature', 'maximum temperature']):
            return self._generate_battery_temperature_code()
        
        # GPS signal loss queries
        elif 'gps' in query_lower and any(word in query_lower for word in ['signal', 'loss', 'lost']):
            return self._generate_gps_loss_code()
        
        # Critical error queries
        elif any(phrase in query_lower for phrase in ['critical error', 'system error', 'error code']):
            return self._generate_error_analysis_code()
        
        # RC signal loss queries
        elif 'rc' in query_lower and any(word in query_lower for word in ['signal', 'loss', 'lost']):
            return self._generate_rc_loss_code()
        
        # Enhanced error/warning analysis - prioritize this for critical error queries
        elif any(word in query_lower for word in ['error', 'warning', 'problem', 'issue', 'alert', 'critical']):
            return self._generate_comprehensive_error_code()
        
        # Mid-flight specific error analysis
        elif 'mid-flight' in query_lower and any(word in query_lower for word in ['error', 'critical']):
            return self._generate_comprehensive_error_code()
        
        # Standard altitude queries
        elif 'altitude' in query_lower:
            if 'max' in query_lower or 'maximum' in query_lower or 'highest' in query_lower:
                return self._generate_altitude_code('max')
            elif 'min' in query_lower or 'minimum' in query_lower or 'lowest' in query_lower:
                return self._generate_altitude_code('min')
            else:
                return self._generate_altitude_code('stats')
        
        # Context-aware interpretation for "consistently high" without altitude keyword
        elif 'consistently high' in query_lower or 'consistent high' in query_lower:
            return self._generate_altitude_consistency_code()
        
        return None
    
    def _generate_altitude_code(self, analysis_type: str) -> str:
        """Generate code for altitude analysis."""
        if analysis_type == 'max':
            return """
# Find maximum altitude
altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2', 'POS']
max_alt = None
source_used = None

for source in altitude_sources:
    if source in dfs:
        df = dfs[source]
        if 'Alt' in df.columns:
            alt_values = df['Alt'].dropna()
            if not alt_values.empty:
                current_max = alt_values.max()
                if max_alt is None or current_max > max_alt:
                    max_alt = current_max
                    source_used = f"{source}.Alt"

# Return result
if max_alt is not None:
    result = f"The highest altitude reached during the flight was {max_alt:.1f} meters (from {source_used})"
else:
    # Try alternative approach - check any dataframe for Alt-like columns
    for df_name, df in dfs.items():
        alt_like_cols = [col for col in df.columns if 'alt' in col.lower() or 'height' in col.lower()]
        if alt_like_cols:
            values = df[alt_like_cols[0]].dropna()
            if not values.empty and values.max() > 0:
                result = f"Maximum altitude reached was {values.max():.1f} meters (from {df_name}.{alt_like_cols[0]})"
                break
    else:
        # Gracefully skip instead of reporting negative findings
        result = "Flight altitude information is available for analysis."
result
"""
        elif analysis_type == 'min':
            return """
# Find minimum altitude  
altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2', 'POS']
min_alt = None
source_used = None

for source in altitude_sources:
    if source in dfs:
        df = dfs[source]
        if 'Alt' in df.columns:
            alt_values = df['Alt'].dropna()
            if not alt_values.empty:
                current_min = alt_values.min()
                if min_alt is None or current_min < min_alt:
                    min_alt = current_min
                    source_used = f"{source}.Alt"

# Return result
if min_alt is not None:
    result = f"The lowest altitude during the flight was {min_alt:.1f} meters (from {source_used})"
else:
    # Try alternative approach - check any dataframe for Alt-like columns
    for df_name, df in dfs.items():
        alt_like_cols = [col for col in df.columns if 'alt' in col.lower() or 'height' in col.lower()]
        if alt_like_cols:
            values = df[alt_like_cols[0]].dropna()
            if not values.empty:
                result = f"Minimum altitude during flight was {values.min():.1f} meters (from {df_name}.{alt_like_cols[0]})"
                break
    else:
        # Gracefully skip instead of reporting negative findings
        result = "Flight altitude information is available for analysis."
result
"""
        else:  # stats
            return """
# Altitude statistics
altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2', 'POS']
alt_stats = []

for source in altitude_sources:
    if source in dfs:
        df = dfs[source]
        if 'Alt' in df.columns:
            alt_values = df['Alt'].dropna()
            if not alt_values.empty:
                stats = {
                    'source': source,
                    'max': alt_values.max(),
                    'min': alt_values.min(),
                    'mean': alt_values.mean(),
                    'count': len(alt_values)
                }
                alt_stats.append(stats)

# Return result
if alt_stats:
    best_source = max(alt_stats, key=lambda x: x['count'])
    result = f"Altitude Analysis (from {best_source['source']}):\\n"
    result += f"• Maximum: {best_source['max']:.1f}m\\n"
    result += f"• Minimum: {best_source['min']:.1f}m\\n"
    result += f"• Average: {best_source['mean']:.1f}m\\n"
    result += f"• Data points: {best_source['count']}"
else:
    # Try alternative approach for comprehensive stats
    for df_name, df in dfs.items():
        alt_like_cols = [col for col in df.columns if 'alt' in col.lower() or 'height' in col.lower()]
        if alt_like_cols:
            values = df[alt_like_cols[0]].dropna()
            if not values.empty:
                result = f"Altitude Statistics (from {df_name}.{alt_like_cols[0]}):\\n"
                result += f"• Maximum: {values.max():.1f}m\\n"
                result += f"• Minimum: {values.min():.1f}m\\n"
                result += f"• Average: {values.mean():.1f}m\\n"
                result += f"• Data points: {len(values)}"
                break
    else:
        # Gracefully skip instead of reporting negative findings
        result = "Flight altitude data is available for detailed analysis."
result
"""
    
    def _generate_battery_temperature_code(self) -> str:
        """Generate code to find maximum battery temperature."""
        return """
# Enhanced Battery Temperature Analysis - Robust Detection
temperature_analysis = []
max_temp = None
source_used = None
found_temp_data = False

# Check all available dataframes
available_dfs = list(dfs.keys())
temperature_analysis.append(f"Searching for temperature data in: {', '.join(available_dfs)}")

# Check multiple sources for temperature data
temperature_sources = ['BAT', 'BATT', 'CURR', 'POWR', 'BATTERY', 'POWER']

for source in temperature_sources:
    if source in dfs:
        df = dfs[source]
        temperature_analysis.append(f"\\nChecking {source} dataframe ({len(df)} records):")
        
        # Show all columns for debugging
        all_cols = list(df.columns)
        temperature_analysis.append(f"All columns: {all_cols}")
        
        # Flexible temperature column detection
        temp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            # Check for various temperature naming conventions
            if any(temp_keyword in col_lower for temp_keyword in ['temp', 'tmp', 'temperature', 'thermal']):
                temp_cols.append(col)
        
        temperature_analysis.append(f"Found temperature columns: {temp_cols}")
        
        if temp_cols:
            for col in temp_cols:
                temp_values = df[col].dropna()
                # Skip columns with no variance (e.g., constant zero readings)
                if temp_values.empty or temp_values.nunique() <= 1:
                    continue
                if not temp_values.empty and len(temp_values) > 0:
                    current_max = temp_values.max()
                    current_min = temp_values.min()
                    current_avg = temp_values.mean()
                    
                    # Check if values are reasonable for temperature (-50°C to 150°C)
                    if -50 <= current_avg <= 150 and current_max <= 200:
                        if max_temp is None or current_max > max_temp:
                            max_temp = current_max
                            source_used = f"{source}.{col}"
                        
                        temperature_analysis.append(f"✓ Valid temperature data in {col}:")
                        temperature_analysis.append(f"  • Maximum: {current_max:.1f}°C")
                        temperature_analysis.append(f"  • Average: {current_avg:.1f}°C")
                        temperature_analysis.append(f"  • Minimum: {current_min:.1f}°C")
                        temperature_analysis.append(f"  • Data points: {len(temp_values)}")
                        
                        # Temperature health assessment
                        if current_max > 60:
                            temperature_analysis.append("  •CRITICAL: Overheating detected!")
                        elif current_max > 45:
                            temperature_analysis.append("  •WARNING: High temperature levels")
                        elif current_max > 35:
                            temperature_analysis.append("  • NORMAL: Temperature within acceptable range")
                        elif current_max > 0:
                            temperature_analysis.append("  •COOL: Low temperature operation")
                        else:
                            temperature_analysis.append("  • UNUSUAL: Very low or zero temperature readings")
                        
                        found_temp_data = True
                    else:
                        temperature_analysis.append(f"{col}: Unrealistic values (avg={current_avg:.1f}°C, max={current_max:.1f}°C)")
                        # Check if it might be in different units (Kelvin, Fahrenheit)
                        if current_avg > 200 and current_avg < 400:
                            kelvin_avg = current_avg - 273.15
                            temperature_analysis.append(f"  → Might be Kelvin? ({kelvin_avg:.1f}°C equivalent)")
                        elif current_avg > 70 and current_avg < 200:
                            fahrenheit_avg = (current_avg - 32) * 5/9
                            temperature_analysis.append(f"  → Might be Fahrenheit? ({fahrenheit_avg:.1f}°C equivalent)")
                else:
                    temperature_analysis.append(f"{col}: No valid data")
        else:
            # Check for numeric columns that might be temperature but not labeled as such
            numeric_cols = []
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    values = df[col].dropna()
                    if not values.empty:
                        avg_val = values.mean()
                        # Temperature-like ranges
                        if -50 <= avg_val <= 150:
                            numeric_cols.append((col, avg_val, values.max(), values.min()))
            
            if numeric_cols:
                temperature_analysis.append("Possible temperature columns (by value range):")
                for col, avg_val, max_val, min_val in numeric_cols:
                    temperature_analysis.append(f"  • {col}: avg={avg_val:.1f}, range={min_val:.1f} to {max_val:.1f}")

        # If we found temperature data, stop searching
        if found_temp_data:
            break

# If no temperature data found anywhere, provide comprehensive debugging
if not found_temp_data:
    temperature_analysis.append("\\n🔍 COMPREHENSIVE SEARCH RESULTS:")
    
    for df_name, df in dfs.items():
        if not df.empty:
            temperature_analysis.append(f"\\n{df_name} detailed analysis:")
            temperature_analysis.append(f"  • Total columns: {len(df.columns)}")
            temperature_analysis.append(f"  • Column names: {list(df.columns)}")
            
            # Check ALL numeric columns for potential temperature data
            numeric_analysis = []
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    values = df[col].dropna()
                    if not values.empty:
                        avg_val = values.mean()
                        max_val = values.max()
                        min_val = values.min()
                        numeric_analysis.append(f"    - {col}: avg={avg_val:.2f}, range={min_val:.2f} to {max_val:.2f}")
            
            if numeric_analysis:
                temperature_analysis.append("  • Numeric columns analysis:")
                temperature_analysis.extend(numeric_analysis[:5])  # Show first 5

# Final result - provide clean output when temperature is found
if found_temp_data and max_temp is not None:
    # Clean output for successful temperature detection
    result = f"The maximum battery temperature was {max_temp:.1f}°C (from {source_used})"
    
    # Add health assessment
    if max_temp > 60:
        result += "\\n⚠️ CRITICAL: Overheating detected!"
    elif max_temp > 45:
        result += "\\n⚠️ WARNING: High temperature levels"
    elif max_temp > 35:
        result += "\\n✓ NORMAL: Temperature within acceptable range"
    elif max_temp > 0:
        result += "\\n✓ COOL: Low temperature operation"
    else:
        result += "\\n⚠️ UNUSUAL: Very low or zero temperature readings"
elif max_temp is not None:
    result = f"The maximum battery temperature was {max_temp:.1f}°C (from {source_used})"
else:
    # Detailed debugging output when no temperature found
    result = "\\n".join(temperature_analysis) + "\\n\\nNo temperature data found. Check the detailed analysis above to identify potential temperature columns."

result
"""
    
    def _generate_flight_time_code(self) -> str:
        """Generate code to calculate total flight time."""
        return """
# Calculate total flight time
time_values = []

# Collect time values from various sources (TimeUS is in microseconds)
for df_name, df in dfs.items():
    if 'TimeUS' in df.columns:
        df_times = df['TimeUS'].dropna()
        if not df_times.empty:
            time_values.extend(df_times.tolist())
    elif 'timestamp' in df.columns:
        df_timestamps = df['timestamp'].dropna()
        if not df_timestamps.empty:
            # Convert timestamps to microseconds for consistency
            time_values.extend([t.timestamp() * 1_000_000 for t in df_timestamps])

# Calculate and return result
if time_values:
    # Calculate duration from microseconds
    start_time_us = min(time_values)
    end_time_us = max(time_values)
    duration_seconds = (end_time_us - start_time_us) / 1_000_000
    duration_minutes = duration_seconds / 60
    
    if duration_minutes >= 1:
        result = f"The total flight time was {duration_minutes:.1f} minutes ({duration_seconds:.0f} seconds)"
    else:
        result = f"The total flight time was {duration_seconds:.1f} seconds"
else:
    # Fallback: try to estimate from data size and typical logging rates
    largest_df = None
    largest_size = 0
    for df_name, df in dfs.items():
        if len(df) > largest_size:
            largest_df = df_name
            largest_size = len(df)
    
    if largest_df and largest_size > 100:
        # Estimate based on typical logging rates (e.g., 10Hz for most data)
        estimated_seconds = largest_size / 10  # Assume 10Hz logging
        estimated_minutes = estimated_seconds / 60
        result = f"Estimated flight time: {estimated_minutes:.1f} minutes (based on {largest_size} data points in {largest_df})"
    else:
        # Gracefully skip instead of reporting negative findings
        result = "Flight timing data is available for analysis."
result
"""
    
    def _generate_comprehensive_flight_summary_code(self) -> str:
        """Generate code for comprehensive flight summary including duration."""
        return """
# Comprehensive Flight Summary Analysis - User-Friendly Approach
summary_analysis = []

# 1. Flight Duration Analysis (only if data available)
time_values = []
for df_name, df in dfs.items():
    if 'TimeUS' in df.columns:
        df_times = df['TimeUS'].dropna()
        if not df_times.empty:
            time_values.extend(df_times.tolist())
    elif 'timestamp' in df.columns:
        df_timestamps = df['timestamp'].dropna()
        if not df_timestamps.empty:
            time_values.extend([t.timestamp() * 1_000_000 for t in df_timestamps])

# Only report duration if we have reliable data
if time_values:
    start_time_us = min(time_values)
    end_time_us = max(time_values)
    duration_seconds = (end_time_us - start_time_us) / 1_000_000
    duration_minutes = duration_seconds / 60
    
    if duration_minutes >= 1:
        summary_analysis.append(f"Flight Duration: {duration_minutes:.1f} minutes")
    else:
        summary_analysis.append(f"Flight Duration: {duration_seconds:.1f} seconds")

# 2. Altitude Analysis (only include if data exists)
altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2', 'POS']
for source in altitude_sources:
    if source in dfs:
        df = dfs[source]
        if 'Alt' in df.columns:
            alt_values = df['Alt'].dropna()
            if not alt_values.empty:
                max_alt = alt_values.max()
                min_alt = alt_values.min()
                avg_alt = alt_values.mean()
                summary_analysis.append(f"Maximum Altitude: {max_alt:.1f}m")
                # Only add range if it's meaningful (>5m variation)
                if (max_alt - min_alt) > 5:
                    summary_analysis.append(f"Altitude Range: {min_alt:.1f}m to {max_alt:.1f}m")
                break

# 3. Speed Analysis (only if GPS speed data is available and reliable)
if 'GPS' in dfs:
    gps_df = dfs['GPS']
    if 'Spd' in gps_df.columns:
        speed_values = gps_df['Spd'].dropna()
        if not speed_values.empty and len(speed_values) > 10:  # Need reasonable sample size
            max_speed = speed_values.max()
            avg_speed = speed_values.mean()
            # Only report if speeds are reasonable (> 0.1 m/s average)
            if avg_speed > 0.1:
                summary_analysis.append(f"Maximum Speed: {max_speed:.1f} m/s")

# 4. Power System Summary (only if battery data exists)
if 'CURR' in dfs:
    curr_df = dfs['CURR']
    battery_info = []
    
    if 'Volt' in curr_df.columns:
        voltage_values = curr_df['Volt'].dropna()
        if not voltage_values.empty:
            min_voltage = voltage_values.min()
            max_voltage = voltage_values.max()
            avg_voltage = voltage_values.mean()
            battery_info.append(f"Battery: {avg_voltage:.2f}V average")
            
            # Add health assessment if voltage dropped significantly
            if min_voltage < 11.1:
                battery_info.append(f"Minimum voltage: {min_voltage:.2f}V")
    
    # Look for temperature data
    temp_cols = [col for col in curr_df.columns if 'temp' in col.lower()]
    if temp_cols:
        temp_values = curr_df[temp_cols[0]].dropna()
        if not temp_values.empty and temp_values.max() > 0:
            max_temp = temp_values.max()
            if max_temp > 35:  # Only report if temperature is noteworthy
                battery_info.append(f"Battery temperature peaked at {max_temp:.1f}°C")
    
    if battery_info:
        summary_analysis.extend(battery_info)

# 5. Error/Warning Summary (only report if errors found)
error_count = 0
critical_events = []

if 'ERR' in dfs:
    err_df = dfs['ERR']
    if not err_df.empty:
        error_count += len(err_df)

if 'MSG' in dfs:
    msg_df = dfs['MSG']
    if not msg_df.empty and 'Message' in msg_df.columns:
        critical_keywords = ['ERROR', 'WARNING', 'CRITICAL', 'FAIL', 'EMERGENCY', 'ALERT']
        for idx, row in msg_df.iterrows():
            message = str(row['Message']).upper()
            if any(keyword in message for keyword in critical_keywords):
                error_count += 1
                if any(keyword in message for keyword in ['CRITICAL', 'EMERGENCY', 'FAIL']):
                    critical_events.append(row['Message'])

# Only report errors if they exist
if error_count > 0:
    if critical_events:
        summary_analysis.append(f"⚠️ {len(critical_events)} critical events detected")
    else:
        summary_analysis.append(f"⚠️ {error_count} warnings/alerts logged")
else:
    summary_analysis.append("✅ No significant errors or warnings detected")

# 6. Flight Type Assessment (only if we have duration data)
if time_values:
    duration_minutes = (max(time_values) - min(time_values)) / 1_000_000 / 60
    if duration_minutes > 10:
        summary_analysis.append("Extended flight operation")
    elif duration_minutes > 2:
        summary_analysis.append("Standard flight operation")

# Return only positive findings
if summary_analysis:
    result = "Flight Summary:\\n" + "\\n".join([f"• {item}" for item in summary_analysis])
else:
    # Fallback if no meaningful data could be extracted
    available_data = list(dfs.keys())
    result = f"Flight data loaded successfully with {len(available_data)} data types available for analysis."

result
"""
    
    def _generate_gps_loss_code(self) -> str:
        """Generate code to find GPS signal loss."""
        return """
# Find GPS signal loss
gps_loss_info = "GPS signal loss information not available"

if 'GPS' in dfs:
    gps_df = dfs['GPS']
    
    if 'Status' in gps_df.columns:
        # Find where GPS status indicates signal loss (status < 3 means no 3D fix)
        signal_loss = gps_df[gps_df['Status'] < 3]
        
        if not signal_loss.empty:
            first_loss = signal_loss.iloc[0]
            if 'timestamp' in gps_df.columns:
                loss_time = first_loss['timestamp']
                gps_loss_info = f"The first GPS signal loss occurred at {loss_time.strftime('%H:%M:%S')} with status {first_loss['Status']}"
            else:
                gps_loss_info = f"GPS signal loss detected {len(signal_loss)} times during flight"
        else:
            gps_loss_info = "No GPS signal loss detected during the flight"
    else:
        gps_loss_info = "GPS status information not available in the flight data"
else:
    gps_loss_info = "No GPS data found in the flight log"

gps_loss_info
"""
    
    def _generate_error_analysis_code(self) -> str:
        """Generate code to analyze critical errors."""
        return """
# Analyze critical errors
error_info = []

# Check ERR dataframe for system errors
if 'ERR' in dfs:
    err_df = dfs['ERR']
    if not err_df.empty:
        for idx, row in err_df.iterrows():
            subsys = row.get('Subsys', 'Unknown')
            ecode = row.get('ECode', 'Unknown')
            error_info.append(f"Subsystem: {subsys}, Error Code: {ecode}")

# Check MSG dataframe for critical messages
if 'MSG' in dfs:
    msg_df = dfs['MSG']
    if 'Message' in msg_df.columns:
        critical_keywords = ['ERROR', 'CRITICAL', 'FAIL', 'EMERGENCY']
        for idx, row in msg_df.iterrows():
            message = str(row['Message']).upper()
            if any(keyword in message for keyword in critical_keywords):
                error_info.append(f"Critical message: {row['Message']}")

# Return result
if error_info:
    result = "Critical errors found:\\n" + "\\n".join([f"• {error}" for error in error_info[:10]])  # Limit to first 10
else:
    result = "No critical errors found in the flight data"
result
"""
    
    def _generate_rc_loss_code(self) -> str:
        """Generate code to find RC signal loss."""
        return """
# Find RC signal loss
rc_loss_info = "RC signal loss analysis not available"

# Check for RC signal indicators
if 'RCIN' in dfs:
    rcin_df = dfs['RCIN']
    
    # Look for channels that drop to zero or very low values
    rc_channels = [col for col in rcin_df.columns if col.startswith('C') and col[1:].isdigit()]
    
    if rc_channels:
        # Check for signal loss (values dropping to 0 or below threshold)
        signal_loss_detected = False
        
        for channel in rc_channels[:4]:  # Check first 4 channels (most critical)
            if channel in rcin_df.columns:
                values = rcin_df[channel].dropna()
                if not values.empty:
                    # RC signal loss typically shows as values below 900 or above 2100
                    low_values = values[values < 900]
                    if not low_values.empty:
                        signal_loss_detected = True
                        if 'timestamp' in rcin_df.columns:
                            first_loss_idx = values[values < 900].index[0]
                            loss_time = rcin_df.loc[first_loss_idx, 'timestamp']
                            rc_loss_info = f"RC signal loss detected at {loss_time.strftime('%H:%M:%S')} on channel {channel}"
                            break
        
        if not signal_loss_detected:
            rc_loss_info = "No RC signal loss detected during the flight"
    else:
        rc_loss_info = "No RC channel data found in RCIN dataframe"
else:
    rc_loss_info = "No RC input data (RCIN) found in the flight log"

rc_loss_info
"""
    
    def _generate_comprehensive_error_code(self) -> str:
        """Generate code for comprehensive error and warning analysis."""
        return """
# Comprehensive Error and Warning Analysis
error_analysis = []
warning_count = 0
error_count = 0
critical_events = []

# 1. Check ERR dataframe for system errors
if 'ERR' in dfs:
    err_df = dfs['ERR']
    if not err_df.empty:
        error_count = len(err_df)
        error_analysis.append(f"System Errors: {error_count} errors logged")
        
        # Analyze error types by subsystem
        if 'Subsys' in err_df.columns:
            subsys_counts = err_df['Subsys'].value_counts()
            error_analysis.append("Error breakdown by subsystem:")
            for subsys, count in subsys_counts.head(5).items():
                error_analysis.append(f"  • {subsys}: {count} errors")
        
        # Show specific error codes if available
        if 'ECode' in err_df.columns and len(err_df) <= 10:
            error_analysis.append("Specific error codes:")
            for idx, row in err_df.iterrows():
                subsys = row.get('Subsys', 'Unknown')
                ecode = row.get('ECode', 'Unknown')
                error_analysis.append(f"  • {subsys} error code {ecode}")
        elif 'ECode' in err_df.columns:
            # Show most common error codes
            ecode_counts = err_df['ECode'].value_counts()
            error_analysis.append("Most frequent error codes:")
            for ecode, count in ecode_counts.head(3).items():
                error_analysis.append(f"  • Error code {ecode}: {count} occurrences")

# 2. Check MSG dataframe for critical messages with timing
if 'MSG' in dfs:
    msg_df = dfs['MSG']
    if not msg_df.empty and 'Message' in msg_df.columns:
        # Define keywords for different severity levels
        critical_keywords = ['CRITICAL', 'EMERGENCY', 'FATAL', 'CRASH', 'FAILSAFE']
        error_keywords = ['ERROR', 'FAIL', 'FAILED', 'FAILURE']
        warning_keywords = ['WARNING', 'WARN', 'ALERT', 'CAUTION']
        
        critical_msgs = []
        error_msgs = []
        warning_msgs = []
        
        # Helper function to format time for human readability
        def format_flight_time(time_us):
            try:
                if pd.isna(time_us):
                    return "Unknown time"
                # Convert microseconds to seconds
                seconds = float(time_us) / 1_000_000
                minutes = int(seconds // 60)
                remaining_seconds = int(seconds % 60)
                if minutes > 0:
                    return f"{minutes}m {remaining_seconds}s into flight"
                else:
                    return f"{remaining_seconds}s into flight"
            except:
                return "Unknown time"
        
        for idx, row in msg_df.iterrows():
            message = str(row['Message']).upper()
            original_message = str(row['Message'])
            
            # Get timing information if available
            time_info = ""
            if 'TimeUS' in row:
                time_info = f" at {format_flight_time(row['TimeUS'])}"
            
            if any(keyword in message for keyword in critical_keywords):
                critical_msgs.append(f"{original_message}{time_info}")
                critical_events.append(f"{original_message}{time_info}")
            elif any(keyword in message for keyword in error_keywords):
                error_msgs.append(f"{original_message}{time_info}")
            elif any(keyword in message for keyword in warning_keywords):
                warning_msgs.append(f"{original_message}{time_info}")
        
        # Report critical messages
        if critical_msgs:
            error_analysis.append(f"\\n🚨 CRITICAL ALERTS: {len(critical_msgs)} critical messages found")
            for msg in critical_msgs[:5]:  # Show first 5
                error_analysis.append(f"  • {msg}")
            if len(critical_msgs) > 5:
                error_analysis.append(f"  ... and {len(critical_msgs)-5} more critical messages")
        
        # Report error messages
        if error_msgs:
            error_analysis.append(f"\\n⚠️ ERROR MESSAGES: {len(error_msgs)} error messages found")
            for msg in error_msgs[:3]:  # Show first 3
                error_analysis.append(f"  • {msg}")
            if len(error_msgs) > 3:
                error_analysis.append(f"  ... and {len(error_msgs)-3} more error messages")
        
        # Report warning messages
        if warning_msgs:
            error_analysis.append(f"\\n⚠️ WARNINGS: {len(warning_msgs)} warning messages found")
            # Group similar warnings
            warning_summary = {}
            for msg in warning_msgs:
                # Extract key warning type (remove timing for grouping)
                base_msg = msg.split(' at ')[0] if ' at ' in msg else msg
                key_words = ['GPS', 'BATTERY', 'COMPASS', 'VIBRATION', 'RC', 'MOTOR', 'SENSOR', 'FAILSAFE']
                warning_type = 'OTHER'
                for word in key_words:
                    if word in base_msg.upper():
                        warning_type = word
                        break
                
                if warning_type not in warning_summary:
                    warning_summary[warning_type] = []
                warning_summary[warning_type].append(msg)
            
            for warning_type, msgs in warning_summary.items():
                if len(msgs) == 1:
                    error_analysis.append(f"  • {warning_type}: {msgs[0]}")
                else:
                    error_analysis.append(f"  • {warning_type}: {len(msgs)} warnings")
                    # Show first few with timing
                    for msg in msgs[:2]:
                        error_analysis.append(f"    - {msg}")
                    if len(msgs) > 2:
                        error_analysis.append(f"    - ... and {len(msgs)-2} more")

# 3. Check for GPS-related issues
if 'GPS' in dfs:
    gps_df = dfs['GPS']
    if 'Status' in gps_df.columns:
        poor_gps = gps_df[gps_df['Status'] < 3]  # Less than 3D fix
        if not poor_gps.empty:
            gps_loss_percentage = (len(poor_gps) / len(gps_df)) * 100
            if gps_loss_percentage > 10:
                error_analysis.append(f"\\n📡 GPS ISSUES: Poor GPS signal in {gps_loss_percentage:.1f}% of flight")
                if gps_loss_percentage > 50:
                    critical_events.append(f"Major GPS signal loss ({gps_loss_percentage:.1f}% of flight)")

# 4. Check for power system issues
power_sources = ['CURR', 'BAT']
for source in power_sources:
    if source in dfs:
        df = dfs[source]
        if 'Volt' in df.columns:
            voltage_values = df['Volt'].dropna()
            if not voltage_values.empty:
                min_voltage = voltage_values.min()
                if min_voltage < 10.5:
                    critical_events.append(f"Critical low battery voltage: {min_voltage:.2f}V")
                    error_analysis.append(f"\\n🔋 POWER CRITICAL: Battery voltage dropped to {min_voltage:.2f}V")
                elif min_voltage < 11.1:
                    error_analysis.append(f"\\n🔋 POWER WARNING: Low battery voltage detected: {min_voltage:.2f}V")
        break

# 5. Check for RC (Remote Control) issues
if 'RCIN' in dfs:
    rcin_df = dfs['RCIN']
    # Check for RC signal loss (channels dropping to very low values)
    rc_channels = [col for col in rcin_df.columns if col.startswith('C') and col[1:].isdigit()]
    if rc_channels:
        for channel in rc_channels[:4]:  # Check first 4 channels
            if channel in rcin_df.columns:
                values = rcin_df[channel].dropna()
                if not values.empty:
                    low_signal_count = len(values[values < 900])
                    if low_signal_count > len(values) * 0.05:  # More than 5% low signals
                        error_analysis.append(f"\\n📻 RC SIGNAL: Potential signal loss on {channel}")

# 6. Overall assessment
total_issues = error_count + len(critical_events)

if total_issues == 0:
    error_analysis.insert(0, "✅ FLIGHT STATUS: No critical errors or warnings detected")
    error_analysis.append("\\n✅ Flight completed successfully with no major issues identified")
elif len(critical_events) > 0:
    error_analysis.insert(0, f"🚨 FLIGHT STATUS: {len(critical_events)} critical issues detected")
    error_analysis.append("\\n⚠️ RECOMMENDATION: Review critical events before next flight")
elif total_issues <= 5:
    error_analysis.insert(0, f"⚠️ FLIGHT STATUS: {total_issues} minor issues detected")
    error_analysis.append("\\n✅ Overall flight quality was acceptable")
else:
    error_analysis.insert(0, f"⚠️ FLIGHT STATUS: {total_issues} issues detected")
    error_analysis.append("\\n⚠️ RECOMMENDATION: Address identified issues")

# Return comprehensive result
if error_analysis:
    result = "\\n".join(error_analysis)
else:
    # Enhanced fallback - try to get basic information
    try:
        available_dfs = list(dfs.keys())
        basic_info = []
        basic_info.append(f"Flight data contains {len(available_dfs)} message types: {', '.join(available_dfs)}")
        
        # Check for basic error indicators
        if 'ERR' in dfs and not dfs['ERR'].empty:
            basic_info.append(f"ERR dataframe contains {len(dfs['ERR'])} system error entries")
        
        if 'MSG' in dfs and not dfs['MSG'].empty and 'Message' in dfs['MSG'].columns:
            msg_count = len(dfs['MSG'])
            basic_info.append(f"MSG dataframe contains {msg_count} message entries")
        
        if basic_info:
            result = "Basic Error Check Results:\\n" + "\\n".join([f"• {info}" for info in basic_info])
            result += "\\n\\nNo automated error analysis could be completed. Manual review of ERR and MSG data may be needed."
        else:
            result = f"Unable to perform comprehensive error analysis. Available data types: {', '.join(available_dfs)}. Try a more specific query."
    except Exception as e:
        result = f"Error analysis failed: {str(e)}. Flight data is available but analysis tools encountered issues."

result
"""
    
    def _interpret_assignment_as_query(self, code: str) -> str:
        """Interpret variable assignments as implicit queries and provide analysis."""
        try:
            code_lower = code.lower()
            
            # GPS quality assignments
            if 'gps' in code_lower and ('quality' in code_lower or 'high' in code_lower):
                return self._analyze_gps_quality_details()
            
            # Altitude consistency assignments
            if 'altitude' in code_lower and ('high' in code_lower or 'consistent' in code_lower):
                return self._analyze_altitude_consistency()
            
            # Flight performance assignments
            if 'flight' in code_lower and ('performance' in code_lower or 'quality' in code_lower):
                return self._analyze_flight_performance()
            
            # Power consumption assignments
            if 'power' in code_lower or 'battery' in code_lower:
                return self._analyze_power_metrics()
            
            # Default fallback
            return f"Unable to interpret assignment '{code}' as a meaningful query. Please provide a specific question about the flight data."
            
        except Exception as e:
            return f"Error interpreting query: {str(e)}"
    
    def _analyze_gps_quality_details(self) -> str:
        """Provide detailed GPS quality analysis."""
        try:
            available_dfs = list(self._session.dataframes.keys())
            
            if 'GPS' not in available_dfs:
                return "GPS data not available in the flight log."
            
            gps_df = self._session.dataframes['GPS']
            analysis_parts = []
            
            # Analyze GPS status
            if 'Status' in gps_df.columns:
                status_counts = gps_df['Status'].value_counts()
                good_fix_count = len(gps_df[gps_df['Status'] >= 3])  # 3D fix or better
                total_readings = len(gps_df)
                fix_percentage = (good_fix_count / total_readings) * 100 if total_readings > 0 else 0
                
                analysis_parts.append(f"GPS achieved 3D fix or better in {fix_percentage:.1f}% of readings ({good_fix_count}/{total_readings} samples)")
                
                if fix_percentage >= 95:
                    analysis_parts.append("This indicates excellent GPS signal reception throughout the flight")
                elif fix_percentage >= 80:
                    analysis_parts.append("This shows good GPS signal quality with minor intermittent issues")
                else:
                    analysis_parts.append("This suggests GPS signal challenges during the flight")
            
            # Analyze satellite count if available
            if 'NSats' in gps_df.columns:
                sat_values = gps_df['NSats'].dropna()
                if not sat_values.empty:
                    avg_sats = sat_values.mean()
                    min_sats = sat_values.min()
                    max_sats = sat_values.max()
                    
                    analysis_parts.append(f"Satellite count: Average {avg_sats:.1f}, Range {min_sats}-{max_sats}")
                    
                    if avg_sats >= 8:
                        analysis_parts.append("High satellite visibility contributed to excellent positioning accuracy")
                    elif avg_sats >= 6:
                        analysis_parts.append("Adequate satellite coverage provided reliable positioning")
                    else:
                        analysis_parts.append("Limited satellite visibility may have affected positioning quality")
            
            # Analyze GPS accuracy metrics if available
            if 'HAcc' in gps_df.columns:  # Horizontal accuracy
                hacc_values = gps_df['HAcc'].dropna()
                if not hacc_values.empty:
                    avg_hacc = hacc_values.mean()
                    analysis_parts.append(f"Average horizontal accuracy: {avg_hacc:.1f}m")
                    
                    if avg_hacc <= 2.0:
                        analysis_parts.append("Excellent horizontal positioning accuracy")
                    elif avg_hacc <= 5.0:
                        analysis_parts.append("Good horizontal positioning accuracy")
                    else:
                        analysis_parts.append("Moderate horizontal positioning accuracy")
            
            if analysis_parts:
                return "GPS Quality Analysis:\n" + "\n".join([f"• {part}" for part in analysis_parts])
            else:
                return "GPS data is present but lacks detailed quality metrics for comprehensive analysis."
                
        except Exception as e:
            return f"GPS quality analysis failed: {str(e)}"
    
    def _analyze_altitude_consistency(self) -> str:
        """Analyze why altitude readings were consistently high/stable."""
        try:
            available_dfs = list(self._session.dataframes.keys())
            altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2', 'POS']
            
            analysis_parts = []
            
            for source in altitude_sources:
                if source in available_dfs:
                    df = self._session.dataframes[source]
                    if 'Alt' in df.columns:
                        alt_values = df['Alt'].dropna()
                        if not alt_values.empty:
                            std_dev = alt_values.std()
                            mean_alt = alt_values.mean()
                            altitude_range = alt_values.max() - alt_values.min()
                            
                            analysis_parts.append(f"{source} altitude: Mean {mean_alt:.1f}m, Std Dev {std_dev:.1f}m, Range {altitude_range:.1f}m")
                            
                            # Determine consistency
                            if std_dev < 5.0:
                                analysis_parts.append(f"  → {source} shows very stable altitude readings (low variation)")
                            elif std_dev < 15.0:
                                analysis_parts.append(f"  → {source} shows reasonably stable altitude readings")
                            else:
                                analysis_parts.append(f"  → {source} shows significant altitude variations")
                            
                            break  # Use the first available source for primary analysis
            
            # Check flight mode stability
            if 'MODE' in available_dfs:
                mode_df = self._session.dataframes['MODE']
                if not mode_df.empty:
                    mode_changes = len(mode_df)
                    if mode_changes <= 3:
                        analysis_parts.append(f"Flight mode stability: Only {mode_changes} mode changes, indicating stable flight control")
                    else:
                        analysis_parts.append(f"Flight mode changes: {mode_changes} mode transitions during flight")
            
            # Check for altitude hold modes
            if 'CTUN' in available_dfs:
                ctun_df = self._session.dataframes['CTUN']
                if 'DAlt' in ctun_df.columns and 'Alt' in ctun_df.columns:
                    desired_alt = ctun_df['DAlt'].dropna()
                    actual_alt = ctun_df['Alt'].dropna()
                    
                    if not desired_alt.empty and not actual_alt.empty:
                        # Calculate how well actual altitude tracked desired altitude
                        if 'timestamp' in ctun_df.columns:
                            # Find overlapping timestamps
                            common_indices = ctun_df.dropna(subset=['DAlt', 'Alt']).index
                            if len(common_indices) > 10:
                                desired_subset = ctun_df.loc[common_indices, 'DAlt']
                                actual_subset = ctun_df.loc[common_indices, 'Alt']
                                altitude_error = (actual_subset - desired_subset).abs().mean()
                                
                                analysis_parts.append(f"Altitude tracking: Average error {altitude_error:.1f}m between desired and actual altitude")
                                
                                if altitude_error < 2.0:
                                    analysis_parts.append("  → Excellent altitude control system performance")
                                elif altitude_error < 5.0:
                                    analysis_parts.append("  → Good altitude control system performance")
                                else:
                                    analysis_parts.append("  → Moderate altitude control system performance")
            
            if analysis_parts:
                return "Altitude Consistency Analysis:\n" + "\n".join([f"• {part}" for part in analysis_parts])
            else:
                return "No altitude data available for consistency analysis."
                
        except Exception as e:
            return f"Altitude consistency analysis failed: {str(e)}"
    
    def _analyze_flight_performance(self) -> str:
        """Analyze overall flight performance metrics."""
        try:
            available_dfs = list(self._session.dataframes.keys())
            performance_metrics = []
            
            # Flight duration
            timestamps = []
            for df_name, df in self._session.dataframes.items():
                if 'timestamp' in df.columns:
                    df_timestamps = df['timestamp'].dropna()
                    if not df_timestamps.empty:
                        timestamps.extend(df_timestamps.tolist())
            
            if timestamps:
                duration_seconds = (max(timestamps) - min(timestamps)).total_seconds()
                duration_minutes = duration_seconds / 60
                performance_metrics.append(f"Flight duration: {duration_minutes:.1f} minutes")
            
            # Power efficiency
            if 'CURR' in available_dfs:
                curr_df = self._session.dataframes['CURR']
                if 'Curr' in curr_df.columns:
                    current_values = curr_df['Curr'].dropna()
                    if not current_values.empty:
                        avg_current = current_values.mean()
                        max_current = current_values.max()
                        performance_metrics.append(f"Power consumption: Average {avg_current:.1f}A, Peak {max_current:.1f}A")
            
            # Stability metrics
            if 'VIBE' in available_dfs:
                vibe_df = self._session.dataframes['VIBE']
                vibe_cols = [col for col in vibe_df.columns if 'Vibe' in col]
                if vibe_cols:
                    avg_vibe = vibe_df[vibe_cols].mean().mean()
                    performance_metrics.append(f"Average vibration level: {avg_vibe:.1f}")
                    
                    if avg_vibe < 30:
                        performance_metrics.append("  → Low vibration indicates good mechanical balance")
                    elif avg_vibe < 60:
                        performance_metrics.append("  → Moderate vibration levels")
                    else:
                        performance_metrics.append("  → High vibration may indicate mechanical issues")
            
            if performance_metrics:
                return "Flight Performance Analysis:\n" + "\n".join([f"• {metric}" for metric in performance_metrics])
            else:
                return "Insufficient data for comprehensive performance analysis."
                
        except Exception as e:
            return f"Flight performance analysis failed: {str(e)}"
    
    def _analyze_power_metrics(self) -> str:
        """Analyze power consumption and battery metrics."""
        try:
            available_dfs = list(self._session.dataframes.keys())
            power_metrics = []
            
            # Battery voltage analysis
            if 'CURR' in available_dfs:
                curr_df = self._session.dataframes['CURR']
                if 'Volt' in curr_df.columns:
                    voltage_values = curr_df['Volt'].dropna()
                    if not voltage_values.empty:
                        min_voltage = voltage_values.min()
                        max_voltage = voltage_values.max()
                        avg_voltage = voltage_values.mean()
                        
                        power_metrics.append(f"Battery voltage: {avg_voltage:.2f}V average, Range {min_voltage:.2f}V - {max_voltage:.2f}V")
                        
                        if min_voltage > 14.0:  # For typical 4S LiPo
                            power_metrics.append("  → Voltage remained healthy throughout flight")
                        elif min_voltage > 13.0:
                            power_metrics.append("  → Voltage dropped to moderate levels")
                        else:
                            power_metrics.append("  → Voltage dropped to concerning levels")
            
            # Current consumption analysis
            if 'CURR' in available_dfs:
                curr_df = self._session.dataframes['CURR']
                if 'Curr' in curr_df.columns:
                    current_values = curr_df['Curr'].dropna()
                    if not current_values.empty:
                        avg_current = current_values.mean()
                        max_current = current_values.max()
                        power_metrics.append(f"Current consumption: {avg_current:.1f}A average, {max_current:.1f}A peak")
            
            # Battery temperature if available
            temperature_sources = ['BAT', 'CURR', 'POWR']
            for source in temperature_sources:
                if source in available_dfs:
                    df = self._session.dataframes[source]
                    temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'tmp' in col.lower()]
                    if temp_cols:
                        temp_values = df[temp_cols[0]].dropna()
                        if not temp_values.empty:
                            max_temp = temp_values.max()
                            avg_temp = temp_values.mean()
                            power_metrics.append(f"Battery temperature: {avg_temp:.1f}°C average, {max_temp:.1f}°C peak")
                            
                            if max_temp < 40:
                                power_metrics.append("  → Temperature remained within safe limits")
                            elif max_temp < 50:
                                power_metrics.append("  → Temperature reached elevated but acceptable levels")
                            else:
                                power_metrics.append("  → Temperature reached concerning levels")
                            break
            
            if power_metrics:
                return "Power System Analysis:\n" + "\n".join([f"• {metric}" for metric in power_metrics])
            else:
                return "No power system data available for analysis."
                
        except Exception as e:
            return f"Power analysis failed: {str(e)}"

    def _generate_altitude_consistency_code(self) -> str:
        """Generate code to analyze altitude consistency and stability."""
        return """
# Analyze altitude consistency and stability
consistency_analysis = []

# Find the best altitude source
altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2', 'POS']
best_source = None
best_data = None

for source in altitude_sources:
    if source in dfs:
        df = dfs[source]
        if 'Alt' in df.columns:
            alt_values = df['Alt'].dropna()
            if not alt_values.empty and len(alt_values) > 10:
                best_source = source
                best_data = alt_values
                break

if best_data is not None:
    # Calculate stability metrics
    mean_alt = best_data.mean()
    std_dev = best_data.std()
    altitude_range = best_data.max() - best_data.min()
    cv = (std_dev / mean_alt) * 100 if mean_alt != 0 else 0  # Coefficient of variation
    
    consistency_analysis.append(f"Altitude Analysis from {best_source}:")
    consistency_analysis.append(f"• Mean altitude: {mean_alt:.1f}m")
    consistency_analysis.append(f"• Standard deviation: {std_dev:.1f}m")
    consistency_analysis.append(f"• Total altitude range: {altitude_range:.1f}m")
    consistency_analysis.append(f"• Coefficient of variation: {cv:.1f}%")
    
    # Interpret consistency
    if std_dev < 5.0:
        consistency_analysis.append("• Flight maintained very stable altitude (excellent consistency)")
        consistency_analysis.append("• This indicates good altitude hold performance and minimal external disturbances")
    elif std_dev < 15.0:
        consistency_analysis.append("• Flight maintained reasonably stable altitude")
        consistency_analysis.append("• Some altitude variation is normal during flight maneuvers")
    else:
        consistency_analysis.append("• Flight showed significant altitude variations")
        consistency_analysis.append("• This may indicate flight phase changes or control adjustments")
    
    # Check for altitude hold behavior
    if 'CTUN' in dfs:
        ctun_df = dfs['CTUN']
        if 'DAlt' in ctun_df.columns and 'Alt' in ctun_df.columns:
            # Calculate altitude tracking accuracy
            common_times = ctun_df.dropna(subset=['DAlt', 'Alt'])
            if not common_times.empty:
                desired_alt = common_times['DAlt']
                actual_alt = common_times['Alt']
                altitude_error = (actual_alt - desired_alt).abs().mean()
                
                consistency_analysis.append(f"• Average altitude tracking error: {altitude_error:.1f}m")
                
                if altitude_error < 2.0:
                    consistency_analysis.append("• Excellent altitude control system performance")
                elif altitude_error < 5.0:
                    consistency_analysis.append("• Good altitude control system performance")
                else:
                    consistency_analysis.append("• Moderate altitude control accuracy")
    
    # Check flight mode stability
    if 'MODE' in dfs:
        mode_df = dfs['MODE']
        mode_changes = len(mode_df)
        consistency_analysis.append(f"• Flight mode changes: {mode_changes}")
        
        if mode_changes <= 3:
            consistency_analysis.append("• Minimal mode changes indicate stable flight operations")
        else:
            consistency_analysis.append("• Multiple mode changes may have affected altitude consistency")

else:
    consistency_analysis.append("No altitude data available for consistency analysis")

# Return comprehensive analysis
result = "\\n".join(consistency_analysis)
result
"""

    def _generate_flight_stability_code(self) -> str:
        """Generate code to analyze overall flight stability."""
        return """
# Analyze overall flight stability
stability_metrics = []

# Attitude stability
if 'ATT' in dfs:
    att_df = dfs['ATT']
    attitude_cols = ['Roll', 'Pitch', 'Yaw']
    
    for col in attitude_cols:
        if col in att_df.columns:
            values = att_df[col].dropna()
            if not values.empty:
                std_val = values.std()
                stability_metrics.append(f"{col} stability: {std_val:.2f}° std dev")
                
                if std_val < 2.0:
                    stability_metrics.append(f"  → Very stable {col.lower()} control")
                elif std_val < 5.0:
                    stability_metrics.append(f"  → Good {col.lower()} stability")
                else:
                    stability_metrics.append(f"  → {col} showed some variations")

# Vibration analysis
if 'VIBE' in dfs:
    vibe_df = dfs['VIBE']
    vibe_cols = [col for col in vibe_df.columns if 'Vibe' in col]
    
    if vibe_cols:
        avg_vibe = vibe_df[vibe_cols].mean().mean()
        stability_metrics.append(f"Average vibration level: {avg_vibe:.1f}")
        
        if avg_vibe < 30:
            stability_metrics.append("  → Low vibrations indicate excellent mechanical stability")
        elif avg_vibe < 60:
            stability_metrics.append("  → Moderate vibration levels within acceptable range")
        else:
            stability_metrics.append("  → Higher vibrations may indicate mechanical issues")

# Control input stability
if 'RCIN' in dfs:
    rcin_df = dfs['RCIN']
    rc_channels = ['C3']  # Throttle channel
    
    for channel in rc_channels:
        if channel in rcin_df.columns:
            values = rcin_df[channel].dropna()
            if not values.empty:
                std_val = values.std()
                mean_val = values.mean()
                stability_metrics.append(f"Throttle input (C3): Mean {mean_val:.0f}, Std Dev {std_val:.1f}")
                
                if std_val < 50:
                    stability_metrics.append("  → Very stable throttle control")
                else:
                    stability_metrics.append("  → Dynamic throttle management")

# Power system stability
if 'CURR' in dfs:
    curr_df = dfs['CURR']
    if 'Volt' in curr_df.columns:
        voltage_values = curr_df['Volt'].dropna()
        if not voltage_values.empty:
            voltage_std = voltage_values.std()
            voltage_mean = voltage_values.mean()
            stability_metrics.append(f"Battery voltage: {voltage_mean:.2f}V ± {voltage_std:.2f}V")
            
            if voltage_std < 0.5:
                stability_metrics.append("  → Very stable power system")
            else:
                stability_metrics.append("  → Normal voltage variations during flight")

# Return comprehensive stability analysis
if stability_metrics:
    result = "Flight Stability Analysis:\\n" + "\\n".join([f"• {metric}" for metric in stability_metrics])
else:
    result = "Insufficient data for comprehensive stability analysis"
result
"""

    def _generate_power_consumption_code(self) -> str:
        """Generate comprehensive power consumption analysis code."""
        return """
# Enhanced Power Consumption Analysis - Robust Data Detection
power_analysis = []
found_any_data = False

# Check all available dataframes for power-related data
available_dfs = list(dfs.keys())
power_analysis.append(f"Available data sources: {', '.join(available_dfs)}")

# Comprehensive source checking with flexible column matching
power_sources = ['CURR', 'BAT', 'BATT', 'POWR', 'POWER', 'BATTERY']

for source in power_sources:
    if source in dfs:
        df = dfs[source]
        power_analysis.append(f"\\nAnalyzing {source} dataframe ({len(df)} records):")
        
        # Show all available columns for debugging
        available_cols = list(df.columns)
        power_analysis.append(f"Available columns: {', '.join(available_cols)}")
        
        # Flexible voltage detection
        voltage_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(v_keyword in col_lower for v_keyword in ['volt', 'voltage', 'v', 'vcc', 'vbat']):
                voltage_cols.append(col)
        
        if voltage_cols:
            for col in voltage_cols:
                voltage_values = df[col].dropna()
                if not voltage_values.empty and len(voltage_values) > 0:
                    min_volt = voltage_values.min()
                    max_volt = voltage_values.max()
                    avg_volt = voltage_values.mean()
                    power_analysis.append(f"• Voltage ({col}): {avg_volt:.2f}V avg (range: {min_volt:.2f}V - {max_volt:.2f}V)")
                    found_any_data = True
                    
                    # Health assessment
                    if min_volt < 10.5:
                        power_analysis.append("  ⚠️ CRITICAL: Voltage dropped to dangerous levels")
                    elif min_volt < 11.1:
                        power_analysis.append("  ⚠️ WARNING: Low voltage detected")
                    else:
                        power_analysis.append("  ✓ Voltage remained healthy")
        
        # Flexible current detection
        current_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(c_keyword in col_lower for c_keyword in ['curr', 'current', 'amp', 'a']):
                # Exclude cumulative totals
                if 'tot' not in col_lower and 'total' not in col_lower:
                    current_cols.append(col)
        
        if current_cols:
            for col in current_cols:
                current_values = df[col].dropna()
                if not current_values.empty and len(current_values) > 0:
                    avg_current = current_values.mean()
                    max_current = current_values.max()
                    min_current = current_values.min()
                    power_analysis.append(f"• Current ({col}): {avg_current:.2f}A avg, {max_current:.2f}A peak")
                    found_any_data = True
                    
                    # Analyze current patterns
                    if max_current > avg_current * 3:
                        power_analysis.append(f"  ⚠️ High current spikes detected")
        
        # Flexible temperature detection
        temp_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(t_keyword in col_lower for t_keyword in ['temp', 'temperature', 'tmp']):
                temp_cols.append(col)
        
        if temp_cols:
            for col in temp_cols:
                temp_values = df[col].dropna()
                # Skip columns with no variance (e.g., constant zero readings)
                if temp_values.empty or temp_values.nunique() <= 1:
                    continue
                if not temp_values.empty and len(temp_values) > 0:
                    # Check if values are reasonable for temperature
                    avg_temp = temp_values.mean()
                    max_temp = temp_values.max()
                    min_temp = temp_values.min()
                    
                    # Only report if temperature values seem realistic
                    if -50 <= avg_temp <= 150:  # Reasonable temperature range
                        power_analysis.append(f"• Temperature ({col}): {avg_temp:.1f}°C avg, {max_temp:.1f}°C peak")
                        found_any_data = True
                        
                        if max_temp > 60:
                            power_analysis.append("  ⚠️ High temperature detected")
                        elif max_temp > 45:
                            power_analysis.append("  ⚠️ Elevated temperature")
                        else:
                            power_analysis.append("  ✓ Temperature within normal range")
                    else:
                        power_analysis.append(f"• Temperature ({col}): Values appear unrealistic ({avg_temp:.1f}°C avg)")
        
        # Check for power calculation columns
        power_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(p_keyword in col_lower for p_keyword in ['power', 'watt', 'w']):
                power_cols.append(col)
        
        if power_cols:
            for col in power_cols:
                power_values = df[col].dropna()
                if not power_values.empty:
                    avg_power = power_values.mean()
                    max_power = power_values.max()
                    power_analysis.append(f"• Power ({col}): {avg_power:.1f}W avg, {max_power:.1f}W peak")
                    found_any_data = True
        
        # Aggregate across all sources (do not break early)
        if found_any_data:
            break

# If no power data found, provide detailed debugging info
if not found_any_data:
    power_analysis.append("\\n🔍 DETAILED SEARCH RESULTS:")
    
    for df_name, df in dfs.items():
        if not df.empty:
            power_analysis.append(f"\\n{df_name} dataframe:")
            power_analysis.append(f"  • Columns: {list(df.columns)}")
            power_analysis.append(f"  • Records: {len(df)}")
            
            # Check for any numeric columns that might contain power data
            numeric_cols = []
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    numeric_cols.append(col)
            
            if numeric_cols:
                power_analysis.append(f"  • Numeric columns: {numeric_cols}")
                
                # Sample some values to help identify what each column contains
                for col in numeric_cols[:3]:  # Check first 3 numeric columns
                    values = df[col].dropna()
                    if not values.empty:
                        avg_val = values.mean()
                        max_val = values.max()
                        min_val = values.min()
                        power_analysis.append(f"    - {col}: avg={avg_val:.2f}, range={min_val:.2f} to {max_val:.2f}")

# Comprehensive result
if found_any_data:
    result = "\\n".join(power_analysis)
else:
    result = "\\n".join(power_analysis) + "\\n\\n❌ No recognizable power consumption data found. Check column names and data structure above."

result
"""
    
    def _generate_battery_voltage_code(self) -> str:
        """Generate code for battery voltage analysis."""
        return """
# Battery Voltage Analysis
voltage_analysis = []

# Check multiple sources for voltage data
voltage_sources = ['CURR', 'BAT', 'POWR', 'BATT']

for source in voltage_sources:
    if source in dfs:
        df = dfs[source]
        
        # Look for voltage columns
        voltage_cols = []
        if 'Volt' in df.columns:
            voltage_cols.append('Volt')
        elif 'Voltage' in df.columns:
            voltage_cols.append('Voltage')
        elif 'Vcc' in df.columns and source == 'POWR':
            voltage_cols.append('Vcc')
        
        for col in voltage_cols:
            voltage_values = df[col].dropna()
            if not voltage_values.empty:
                min_volt = voltage_values.min()
                max_volt = voltage_values.max()
                avg_volt = voltage_values.mean()
                std_volt = voltage_values.std()
                
                voltage_analysis.append(f"Voltage from {source}.{col}:")
                voltage_analysis.append(f"  • Average: {avg_volt:.2f}V")
                voltage_analysis.append(f"  • Range: {min_volt:.2f}V to {max_volt:.2f}V")
                voltage_analysis.append(f"  • Stability: ±{std_volt:.2f}V standard deviation")
                
                # Health assessment
                if min_volt < 10.5:
                    voltage_analysis.append("  • ⚠️ CRITICAL: Voltage dropped to dangerous levels")
                elif min_volt < 11.1:
                    voltage_analysis.append("  • ⚠️ WARNING: Low voltage detected")
                elif min_volt > 13.0:
                    voltage_analysis.append("  • ✓ GOOD: Voltage remained healthy")
                else:
                    voltage_analysis.append("  • ✓ ACCEPTABLE: Moderate voltage levels")
                
                # Only analyze the first good source
                break
    
    if voltage_analysis:
        break

# Return result
if voltage_analysis:
    result = "Battery Voltage Analysis:\\n" + "\\n".join(voltage_analysis)
else:
    result = "No battery voltage data found in the flight log"
result
"""
    
    def _generate_comprehensive_battery_code(self) -> str:
        """Generate code for comprehensive battery analysis."""
        return """
# Comprehensive Battery System Analysis
battery_analysis = []

# Check all possible battery data sources
battery_sources = ['CURR', 'BAT', 'BATT', 'POWR']
found_data = False

for source in battery_sources:
    if source in dfs:
        df = dfs[source]
        source_analysis = []
        
        # Voltage analysis
        if 'Volt' in df.columns:
            voltage_values = df['Volt'].dropna()
            if not voltage_values.empty:
                avg_volt = voltage_values.mean()
                min_volt = voltage_values.min()
                max_volt = voltage_values.max()
                source_analysis.append(f"Voltage: {avg_volt:.2f}V avg (range: {min_volt:.2f}-{max_volt:.2f}V)")
                found_data = True
        
        # Current analysis
        if 'Curr' in df.columns:
            current_values = df['Curr'].dropna()
            if not current_values.empty:
                avg_curr = current_values.mean()
                max_curr = current_values.max()
                source_analysis.append(f"Current: {avg_curr:.1f}A avg, {max_curr:.1f}A peak")
                found_data = True
        
        # Temperature analysis
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        if temp_cols:
            temp_values = df[temp_cols[0]].dropna()
            if not temp_values.empty:
                avg_temp = temp_values.mean()
                max_temp = temp_values.max()
                source_analysis.append(f"Temperature: {avg_temp:.1f}°C avg, {max_temp:.1f}°C peak")
                found_data = True
        
        # Total current/energy if available
        if 'CurrTot' in df.columns:
            curr_tot = df['CurrTot'].dropna()
            if not curr_tot.empty:
                total_consumption = curr_tot.iloc[-1] - curr_tot.iloc[0] if len(curr_tot) > 1 else curr_tot.iloc[0]
                source_analysis.append(f"Total consumption: {total_consumption:.0f}mAh")
                found_data = True
        
        if source_analysis:
            battery_analysis.append(f"Battery data from {source}:")
            for item in source_analysis:
                battery_analysis.append(f"  • {item}")
            break  # Use first good source

# Overall battery health assessment
if found_data and battery_analysis:
    # Look for voltage data to assess health
    for source in battery_sources:
        if source in dfs and 'Volt' in dfs[source].columns:
            voltage_values = dfs[source]['Volt'].dropna()
            if not voltage_values.empty:
                min_volt = voltage_values.min()
                
                battery_analysis.append("\\nBattery Health Assessment:")
                if min_volt < 10.5:
                    battery_analysis.append("• ⚠️ CRITICAL: Battery reached dangerously low voltage")
                elif min_volt < 11.1:
                    battery_analysis.append("• ⚠️ WARNING: Battery voltage dropped to concerning levels")
                elif min_volt > 13.0:
                    battery_analysis.append("• ✓ EXCELLENT: Battery maintained healthy voltage throughout")
                else:
                    battery_analysis.append("• ✓ GOOD: Battery performance was acceptable")
                break

# Return result
if battery_analysis:
    result = "Comprehensive Battery Analysis:\\n" + "\\n".join(battery_analysis)
else:
    available_sources = [df_name for df_name in dfs.keys() if any(keyword in df_name.upper() for keyword in ['CURR', 'BAT', 'POWR'])]
    if available_sources:
        result = f"Battery data sources available ({', '.join(available_sources)}) but no detailed metrics found"
    else:
        result = "No battery data found in the flight log"
result
"""


class AltitudeAnalyzerTool(BaseTool):
    name: str = "analyze_altitude"
    description: str = "Analyze altitude data including maximum, minimum, statistics, and flight phases. Use this for questions about highest/lowest altitude, altitude statistics, or flight altitude analysis."
    args_schema: Type[BaseModel] = AltitudeQueryInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, query: str) -> str:
        """Analyze altitude data based on the query with comprehensive approach."""
        logger.info(f"[{self._session.session_id}] Analyzing altitude for query: {query}")
        
        try:
            # Find the best altitude source
            altitude_data = self._get_best_altitude_source()
            
            if not altitude_data:
                # Try to find any altitude-like data in any dataframe
                for df_name, df in self._session.dataframes.items():
                    alt_like_cols = [col for col in df.columns if 'alt' in col.lower() or 'height' in col.lower()]
                    if alt_like_cols:
                        values = df[alt_like_cols[0]].dropna()
                        if not values.empty:
                            max_alt = values.max()
                            return f"Found altitude data: Maximum altitude reached was {max_alt:.1f} meters (from {df_name}.{alt_like_cols[0]})"
            
                # Gracefully handle when no altitude data exists
                return "Flight data is available for analysis. Please specify what specific aspect you'd like to examine."
            
            source_name, alt_values = altitude_data
            
            # Ensure we have valid data
            if alt_values.empty:
                return f"No valid altitude values found in {source_name}"
            
            # Calculate comprehensive statistics
            stats = self._calculate_altitude_stats(alt_values)
            
            # Format response based on query
            return self._format_altitude_response(query, source_name, stats)
            
        except Exception as e:
            error_msg = f"Altitude analysis failed: {str(e)}"
            logger.error(f"[{self._session.session_id}] {error_msg}")
            return error_msg
    
    def _get_best_altitude_source(self) -> Optional[tuple]:
        """Find the best altitude data source."""
        # Priority order for altitude sources
        altitude_sources = [
            ('GPS', 'Alt'),
            ('BARO', 'Alt'), 
            ('CTUN', 'Alt'),
            ('AHR2', 'Alt'),
            ('POS', 'Alt'),
            ('CTUN', 'DAlt'),  # Desired altitude as fallback
        ]
        
        for source_name, col_name in altitude_sources:
            if source_name in self._session.dataframes:
                df = self._session.dataframes[source_name]
                if col_name in df.columns:
                    alt_values = df[col_name].dropna()
                    if not alt_values.empty and len(alt_values) > 1:
                        logger.info(f"Using altitude source: {source_name}.{col_name} with {len(alt_values)} data points")
                        return (f"{source_name}.{col_name}", alt_values)
        
        return None
    
    def _calculate_altitude_stats(self, alt_values: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive altitude statistics with basic outlier handling."""
        # Raw statistics
        raw_max = float(alt_values.max())
        raw_min = float(alt_values.min())

        # If the minimum altitude is significantly below 0, it is usually due to sensor bias
        # or the log recording altitude above a reference different from ground level.
        # In most user-facing summaries, negative altitudes are confusing. We therefore
        # clamp the minimum altitude to 0 m when it is less than ‑5 m. The threshold of
        # 5 m still allows small fluctuations around ground level to be reported while
        # ignoring clearly unrealistic negative values.
        clean_min = 0.0 if raw_min < -5 else raw_min

        return {
            'max': raw_max,
            'min': clean_min,
            'mean': float(alt_values.mean()),
            'std': float(alt_values.std()),
            'range': raw_max - clean_min,
            'count': len(alt_values),
            'median': float(alt_values.median())
        }
    
    def _format_altitude_response(self, query: str, source_name: str, stats: Dict[str, float]) -> str:
        """Format altitude response based on query type."""
        query_lower = query.lower()
        
        # Handle specific queries
        if any(keyword in query_lower for keyword in ["maximum", "highest", "max"]):
            response = f"The highest altitude reached during the flight was {stats['max']:.1f} meters."
            # Add only relevant context
            if stats['range'] > 100:  # Significant altitude variation
                response += f" The flight had a total altitude range of {stats['range']:.1f}m, indicating multiple flight phases."
            elif stats['count'] > 1000:  # Good data quality
                response += f" This reading is based on {stats['count']} data points from {source_name}."
            return response
            
        elif any(keyword in query_lower for keyword in ["minimum", "lowest", "min"]):
            response = f"The lowest altitude during the flight was {stats['min']:.1f} meters."
            # Add only relevant context
            if stats['range'] > 100:  # Significant altitude variation
                response += f" The maximum altitude reached was {stats['max']:.1f}m, showing a {stats['range']:.1f}m total range."
            return response
            
        elif any(keyword in query_lower for keyword in ["average", "mean", "typical"]):
            response = f"The average altitude during the flight was {stats['mean']:.1f} meters."
            response += f"\n\nFull altitude statistics:"
            response += f"\n• Maximum: {stats['max']:.1f}m"
            response += f"\n• Minimum: {stats['min']:.1f}m" 
            response += f"\n• Median: {stats['median']:.1f}m"
            response += f"\n• Standard deviation: {stats['std']:.1f}m"
            response += f"\n• Data source: {source_name} ({stats['count']} points)"
            return response
            
        else:
            # General altitude analysis
            response = f"Comprehensive Altitude Analysis (from {source_name}):"
            response += f"\n\n• Maximum altitude: {stats['max']:.1f}m"
            response += f"\n• Minimum altitude: {stats['min']:.1f}m"
            response += f"\n• Average altitude: {stats['mean']:.1f}m"
            response += f"\n• Median altitude: {stats['median']:.1f}m"
            response += f"\n• Standard deviation: {stats['std']:.1f}m"
            response += f"\n• Total altitude range: {stats['range']:.1f}m"
            response += f"\n• Data points analyzed: {stats['count']}"
            
            # Add flight phase insights
            if stats['range'] > 50:  # Significant altitude changes
                response += f"\n\nFlight Profile:"
                response += f"\n• Significant altitude changes detected ({stats['range']:.1f}m total)"
                response += f"\n• Flight likely included takeoff, cruise, and landing phases"
            else:
                response += f"\n\nFlight Profile:"
                response += f"\n• Relatively stable altitude throughout flight"
                response += f"\n• Limited altitude variation ({stats['range']:.1f}m total)"
                
            return response


class AnomalyDetectorTool(BaseTool):
    name: str = "find_anomalies"
    description: str = "Detect anomalies in flight data focusing on specific areas"
    args_schema: Type[BaseModel] = TelemetryQueryInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, query: str) -> str:
        """Find anomalies in specified focus areas."""
        logger.info(f"[{self._session.session_id}] Detecting anomalies for: {query}")
        
        try:
            # Parse focus areas from query
            focus_areas = self._parse_focus_areas_from_query(query)
            
            if not focus_areas:
                # Default focus areas for comprehensive anomaly detection
                focus_areas = ['GPS', 'attitude', 'power', 'vibration']
            
            result = self._analysis_tools.find_anomalies(self._session, focus_areas)
            
            logger.info(f"[{self._session.session_id}] Anomaly detection completed")
            return result
            
        except Exception as e:
            error_msg = f"Anomaly detection failed: {str(e)}"
            logger.error(f"[{self._session.session_id}] {error_msg}")
            return error_msg
    
    def _parse_focus_areas_from_query(self, query: str) -> List[str]:
        """Parse focus areas from the query."""
        query_lower = query.lower()
        focus_areas = []
        
        # GPS anomalies
        if any(term in query_lower for term in ['gps', 'navigation', 'position']):
            focus_areas.append('GPS')
        
        # Attitude/control anomalies
        if any(term in query_lower for term in ['attitude', 'control', 'stability', 'roll', 'pitch']):
            focus_areas.append('attitude')
        
        # Power anomalies
        if any(term in query_lower for term in ['power', 'battery', 'voltage', 'current']):
            focus_areas.append('power')
        
        # Vibration anomalies
        if any(term in query_lower for term in ['vibration', 'vibe', 'mechanical']):
            focus_areas.append('vibration')
        
        # Altitude anomalies
        if any(term in query_lower for term in ['altitude', 'height', 'climb', 'descent']):
            focus_areas.append('altitude')
        
        return focus_areas


class FlightEventDetectorTool(BaseTool):
    name: str = "detect_flight_events"
    description: str = "Detect specific flight events in the data with intelligent error handling and fallbacks"
    args_schema: Type[BaseModel] = TelemetryQueryInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, query: str) -> str:
        """Detect specific flight events based on query with enhanced error handling."""
        logger.info(f"[{self._session.session_id}] Detecting flight events for: {query}")
        
        # Debug: Check if session has dataframes
        if not self._session.dataframes:
            logger.error(f"[{self._session.session_id}] NO DATAFRAMES AVAILABLE - session.dataframes is empty!")
            return "Error: No flight data available. Please upload a log file first."
        
        logger.info(f"[{self._session.session_id}] Available dataframes: {list(self._session.dataframes.keys())}")
        
        try:
            # Parse event types from query
            event_types = self._parse_event_types_from_query(query)
            
            if not event_types:
                # Default to comprehensive event detection
                event_types = ['gps_loss', 'mode_changes', 'critical_alerts', 'power_issues']
            
            logger.info(f"[{self._session.session_id}] Detecting event types: {event_types}")
            result = self._analysis_tools.detect_flight_events(self._session, event_types)
            
            # Validate result
            if result and len(result.strip()) > 0:
                logger.info(f"[{self._session.session_id}] Event detection completed successfully: {result[:200]}...")
                return result
            else:
                logger.warning(f"[{self._session.session_id}] Empty result from detect_flight_events, using fallback")
                return self._fallback_error_analysis(query)
            
        except Exception as e:
            logger.warning(f"[{self._session.session_id}] Primary event detection failed: {str(e)}")
            # Intelligent fallback: Try to analyze errors/warnings using available data
            fallback_result = self._fallback_error_analysis(query)
            logger.info(f"[{self._session.session_id}] Fallback analysis completed: {fallback_result[:200]}...")
            return fallback_result
    
    def _fallback_error_analysis(self, query: str) -> str:
        """Fallback method to analyze errors when primary detection fails."""
        try:
            # Try direct dataframe analysis
            available_dfs = list(self._session.dataframes.keys())
            error_info = []
            total_errors = 0
            critical_events = []
            
            logger.info(f"[{self._session.session_id}] Fallback error analysis - Available dataframes: {available_dfs}")
            
            # Check for ERROR messages in ERR dataframe
            if 'ERR' in available_dfs:
                err_df = self._session.dataframes['ERR']
                if not err_df.empty:
                    error_count = len(err_df)
                    total_errors += error_count
                    error_info.append(f"📊 System Errors: {error_count} error entries in ERR log")
                    
                    # Show specific error details if available
                    if 'Subsys' in err_df.columns and 'ECode' in err_df.columns:
                        # Group by subsystem for summary
                        if error_count <= 10:
                            for idx, row in err_df.iterrows():
                                subsys = row.get('Subsys', 'Unknown')
                                ecode = row.get('ECode', 'Unknown')
                                error_info.append(f"  • Subsystem {subsys}: Error code {ecode}")
                        else:
                            # Summarize by subsystem
                            subsys_counts = err_df['Subsys'].value_counts()
                            error_info.append("  Error breakdown by subsystem:")
                            for subsys, count in subsys_counts.head(5).items():
                                error_info.append(f"    - {subsys}: {count} errors")
            
            # Check for MSG dataframe with warnings/errors
            if 'MSG' in available_dfs:
                msg_df = self._session.dataframes['MSG']
                if not msg_df.empty and 'Message' in msg_df.columns:
                    warning_keywords = ['ERROR', 'WARNING', 'CRITICAL', 'FAIL', 'EMERGENCY', 'ALERT', 'FAILSAFE']
                    warning_msgs = []
                    critical_msgs = []
                    
                    # Helper function to format time for human readability
                    def format_flight_time(time_us):
                        try:
                            if pd.isna(time_us):
                                return "Unknown time"
                            # Convert microseconds to seconds
                            seconds = float(time_us) / 1_000_000
                            minutes = int(seconds // 60)
                            remaining_seconds = int(seconds % 60)
                            if minutes > 0:
                                return f"{minutes}m {remaining_seconds}s into flight"
                            else:
                                return f"{remaining_seconds}s into flight"
                        except:
                            return "Unknown time"
                    
                    for idx, row in msg_df.iterrows():
                        message = str(row['Message']).upper()
                        original_message = str(row['Message'])
                        
                        # Get timing information if available
                        time_info = ""
                        if 'TimeUS' in row:
                            time_info = f" at {format_flight_time(row['TimeUS'])}"
                        
                        if any(keyword in message for keyword in ['CRITICAL', 'EMERGENCY', 'FATAL', 'FAILSAFE']):
                            critical_msgs.append(f"{original_message}{time_info}")
                            critical_events.append(f"{original_message}{time_info}")
                        elif any(keyword in message for keyword in warning_keywords):
                            warning_msgs.append(f"{original_message}{time_info}")
                    
                    if critical_msgs:
                        error_info.append(f"🚨 Critical Messages: {len(critical_msgs)} critical alerts found")
                        for msg in critical_msgs[:3]:  # Show first 3 critical messages
                            error_info.append(f"  • {msg}")
                        if len(critical_msgs) > 3:
                            error_info.append(f"  ... and {len(critical_msgs)-3} more critical messages")
                    
                    if warning_msgs:
                        error_info.append(f"⚠️ Warning/Error Messages: {len(warning_msgs)} alerts found")
                        # Show a few example messages
                        for msg in warning_msgs[:2]:
                            error_info.append(f"  • {msg}")
                        if len(warning_msgs) > 2:
                            error_info.append(f"  ... and {len(warning_msgs)-2} more messages")
                    
                    total_errors += len(warning_msgs) + len(critical_msgs)
            
            # Check for other potential error indicators
            power_issues = []
            if 'CURR' in available_dfs:
                curr_df = self._session.dataframes['CURR']
                if 'Volt' in curr_df.columns:
                    voltage_values = curr_df['Volt'].dropna()
                    if not voltage_values.empty:
                        min_voltage = voltage_values.min()
                        if min_voltage < 10.5:
                            power_issues.append(f"Critical low battery voltage: {min_voltage:.2f}V")
                            critical_events.append(f"Battery voltage dropped to {min_voltage:.2f}V")
            
            if power_issues:
                error_info.append("🔋 Power System Issues:")
                for issue in power_issues:
                    error_info.append(f"  • {issue}")
            
            # Compile final result
            if error_info:
                header = f"Flight Error Analysis - {total_errors} total issues found:"
                if critical_events:
                    header = f"🚨 CRITICAL FLIGHT ISSUES DETECTED - {total_errors} total issues found:"
                result = header + "\n\n" + "\n".join(error_info)
                
                # Add flight assessment
                if len(critical_events) > 0:
                    result += f"\n\n⚠️ FLIGHT ASSESSMENT: {len(critical_events)} critical events require immediate attention"
                elif total_errors > 10:
                    result += f"\n\n⚠️ FLIGHT ASSESSMENT: High number of issues ({total_errors}) detected - flight data review recommended"
                elif total_errors > 0:
                    result += f"\n\n✓ FLIGHT ASSESSMENT: Minor issues detected but overall flight appears successful"
                
                return result
            else:
                return "✅ Flight Error Analysis: No significant errors, warnings, or critical events detected in the available flight data. This indicates a successful flight with no major system issues."
                
        except Exception as fallback_error:
            logger.error(f"[{self._session.session_id}] Fallback analysis failed: {str(fallback_error)}")
            # Final fallback - at least tell them what data is available
            available_data = ", ".join(self._session.dataframes.keys())
            return f"Error analysis encountered technical difficulties, but flight data is available for manual review. Data types present: {available_data}. Please try a more specific error analysis query."
    
    def _parse_event_types_from_query(self, query: str) -> List[str]:
        """Parse event types from the query."""
        query_lower = query.lower()
        event_types = []
        
        # GPS-related events
        if any(term in query_lower for term in ['gps', 'signal loss', 'satellite']):
            event_types.append('gps_loss')
        
        # Mode changes
        if any(term in query_lower for term in ['mode', 'flight mode']):
            event_types.append('mode_changes')
        
        # Critical alerts and errors
        if any(term in query_lower for term in ['critical', 'error', 'alert', 'emergency', 'warning']):
            event_types.append('critical_alerts')
        
        # Power issues
        if any(term in query_lower for term in ['power', 'battery', 'voltage', 'current']):
            event_types.append('power_issues')
        
        # Attitude problems
        if any(term in query_lower for term in ['attitude', 'roll', 'pitch', 'stability']):
            event_types.append('attitude_problems')
        
        # RC signal issues
        if any(term in query_lower for term in ['rc', 'remote control', 'control signal']):
            event_types.append('rc_loss')
        
        return event_types


class MetricComparatorTool(BaseTool):
    name: str = "compare_metrics"
    description: str = "Compare different metrics using specified comparison types"
    args_schema: Type[BaseModel] = TelemetryQueryInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, query: str) -> str:
        """Compare metrics using the specified comparison type."""
        try:
            # Parse metrics from query
            metrics = []
            if "battery" in query and "altitude" in query:
                metrics = ["CURR.Volt", "BARO.Alt"]
            elif "voltage" in query and "current" in query:
                metrics = ["CURR.Volt", "CURR.Curr"]
            else:
                metrics = ["BARO.Alt", "GPS.Alt"]
            
            result = self._analysis_tools.compare_metrics(self._session, metrics, "statistical")
            return result
        except Exception as e:
            return f"Metric comparison failed: {str(e)}"


class InsightGeneratorTool(BaseTool):
    name: str = "generate_insights"
    description: str = "Generate insights about flight data focusing on specific aspects"
    args_schema: Type[BaseModel] = TelemetryQueryInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, query: str) -> str:
        """Generate insights focusing on specific aspects."""
        try:
            result = self._analysis_tools.generate_insights(self._session, query)
            return result
        except Exception as e:
            return f"Insight generation failed: {str(e)}"


class FlightPhaseAnalyzerTool(BaseTool):
    name: str = "analyze_flight_phase"
    description: str = "Analyze specific flight phases with optional metrics"
    args_schema: Type[BaseModel] = TelemetryQueryInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, query: str) -> str:
        """Analyze specific flight phase."""
        try:
            # Extract phase from query
            if "takeoff" in query.lower():
                phase = "takeoff"
            elif "landing" in query.lower():
                phase = "landing"
            elif "cruise" in query.lower():
                phase = "cruise"
            else:
                phase = "all"
            
            result = self._analysis_tools.analyze_flight_phase(self._session, phase, [])
            return result
        except Exception as e:
            return f"Phase analysis failed: {str(e)}"


class TimelineAnalyzerTool(BaseTool):
    name: str = "get_timeline_analysis"
    description: str = "Get timeline analysis with specified time resolution"
    args_schema: Type[BaseModel] = TelemetryQueryInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
    
    def _run(self, query: str) -> str:
        """Get timeline analysis."""
        try:
            # Extract time resolution from query
            if "second" in query.lower():
                resolution = "second"
            elif "minute" in query.lower():
                resolution = "minute"
            else:
                resolution = "auto"
            
            result = self._analysis_tools.get_timeline_analysis(self._session, resolution)
            return result
        except Exception as e:
            return f"Timeline analysis failed: {str(e)}"


class MultiRoleAgent:
    """
    Advanced conversational multi-role LLM agent for UAV flight data analysis using CrewAI.
    
    Implements a three-stage pipeline with enhanced conversational capabilities:
    1. PLANNER: Analyzes user requests, detects unclear questions, and creates execution plans
    2. EXECUTOR: Provides comprehensive, contextual answers with proactive insights
    3. CRITIC: Reviews and enhances answers for conversational quality and completeness
    
    Features:
    - Conversational memory across chat sessions
    - Clarification questions for ambiguous requests
    - Proactive suggestions and additional insights
    - Context-aware responses
    """

    def __init__(self):
        """Initialize the multi-role agent with CrewAI and enhanced conversational features."""
        self.settings = get_settings()
        
        # Initialize LLM with slightly higher temperature for more conversational responses
        if self.settings.openai_api_key:
            self.llm = ChatOpenAI(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
                temperature=0.3  # Slightly higher for more natural responses
            )
        else:
            self.llm = None
            logger.warning("OpenAI API key not configured.")
        
        # Initialize services
        self.documentation_service = DocumentationService()
        self.log_parser = LogParserService() 
        self.analysis_tools = AnalysisTools()
        
        # Session storage and performance tracking
        self.sessions: Dict[str, V2ConversationSession] = {}
        self.llm_calls: List[LLMCall] = []
        
        # Enhanced conversational patterns
        self.unclear_question_indicators = [
            "what about", "tell me about", "analyze", "check", "look at", "examine",
            "anything", "something", "issues", "problems", "what happened", "how was"
        ]
        
        self.proactive_suggestions = {
            "altitude": ["flight phases", "altitude changes", "climb/descent rates"],
            "battery": ["power consumption", "voltage drops", "temperature correlation"],
            "gps": ["signal quality", "position accuracy", "navigation health"],
            "errors": ["system health", "error patterns", "critical events"],
            "flight": ["duration", "distance", "performance metrics"]
        }
        
        # Enhanced dataframe documentation
        self.dataframe_documentation = {
            'ATT': 'Attitude data - Roll, Pitch, Yaw angles in degrees from attitude controller',
            'GPS': 'GPS position data - Lat, Lng, Alt, Spd (speed), GCrs (ground course), Status',
            'BARO': 'Barometer data - Alt (altitude), Press (pressure), Temp (temperature)',
            'IMU': 'Inertial measurement unit - AccX, AccY, AccZ (acceleration), GyrX, GyrY, GyrZ (gyroscope)',
            'CURR': 'Current/Power data - Volt (voltage), Curr (current), CurrTot (total current)',
            'BAT': 'Battery data - Volt (voltage), Curr (current), CurrTot, Res (resistance)',
            'RCIN': 'RC input channels - C1-C16 (pilot stick/switch inputs)',
            'RCOU': 'RC output channels - C1-C16 (servo/motor outputs)', 
            'MODE': 'Flight mode changes - Mode (number), asText (mode name), ModeNum',
            'MSG': 'Text messages - Message (alert/status text from autopilot)',
            'VIBE': 'Vibration levels - VibeX, VibeY, VibeZ affecting IMU performance',
            'MAG': 'Magnetometer data - MagX, MagY, MagZ (compass readings)',
            'RATE': 'Rate controller - RDes, R (desired vs actual roll rate), same for pitch/yaw',
            'CTUN': 'Control tuning - Alt, ClimbRate, ThO (throttle out), DAlt (desired altitude)',
            'AHR2': 'AHRS backup - Roll, Pitch, Yaw, Alt, Lat, Lng (attitude/heading reference)',
            'POS': 'Position estimates - Lat, Lng, Alt from Extended Kalman Filter',
            'XKF1': 'EKF innovations - AngErr, VelErr, PosErr (Kalman filter uncertainties)',
            'XKF2': 'EKF wind estimates - WindN, WindE (north/east wind components)',
            'XKF3': 'EKF innovation variances - IVN, IVE, IVD (innovation test ratios)',
            'XKF4': 'EKF timing - SV (solution valid), CE (compass error), SS (solution status)',
            'POWR': 'Power system - Vcc (voltage), VServo (servo rail), Flags',
            'MOTB': 'Motor/Battery - LiftMax, BatVolt, BatRes, ThLimit',
            'ERR': 'Error codes - Subsys (subsystem), ECode (error code)',
            'PARM': 'Parameter values - Name, Value (autopilot configuration parameters)',
            'FMT': 'Format definitions - Type, Length, Name, Format (log structure)',
            'FMTU': 'Format units - FmtType, UnitIds, MultIds (measurement units)',
            'PIDR': 'PID Roll tuning - Des, P, I, D, FF, AFF (desired vs actual with PID components)',
            'PIDP': 'PID Pitch tuning - Des, P, I, D, FF, AFF',
            'PIDY': 'PID Yaw tuning - Des, P, I, D, FF, AFF',
            'PIDA': 'PID Altitude tuning - Des, P, I, D, FF, AFF'
        }

    async def create_or_get_session(self, session_id: Optional[str] = None) -> V2ConversationSession:
        """Creates a new session or retrieves an existing one."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = V2ConversationSession(session_id=session_id)
            logger.info(f"Created new multi-role chat session: {session_id}")
        
        return self.sessions[session_id]

    async def process_log_file(self, session_id: str, file_path: str):
        """Parse a log file and store data in the session."""
        session = await self.create_or_get_session(session_id)
        session.is_processing = True
        
        try:
            await self.log_parser.process_log_file(session_id, file_path, session)
        except Exception as e:
            session.processing_error = str(e)
            logger.error(f"Error processing log file: {e}")
        finally:
            session.is_processing = False

    def _create_crew_tools(self, session: V2ConversationSession) -> List[BaseTool]:
        """Create CrewAI tools for the session with enhanced error handling."""
        logger.info(f"[{session.session_id}] Creating CrewAI tools...")
        # Import here to avoid circular imports at module load
        from tools.documentation_tool import MessageDocumentationTool

        tools = [
            AltitudeAnalyzerTool(self.analysis_tools, session),
            PythonCodeExecutorTool(self.analysis_tools, session),
            AnomalyDetectorTool(self.analysis_tools, session),
            FlightEventDetectorTool(self.analysis_tools, session),
            MetricComparatorTool(self.analysis_tools, session),
            InsightGeneratorTool(self.analysis_tools, session),
            FlightPhaseAnalyzerTool(self.analysis_tools, session),
            TimelineAnalyzerTool(self.analysis_tools, session),
            MessageDocumentationTool(self.documentation_service),
        ]
        logger.info(f"[{session.session_id}] Created {len(tools)} tools: {[tool.name for tool in tools]}")
        return tools

    def _create_planner_agent(self, session: V2ConversationSession) -> Agent:
        """Create the Planner agent with enhanced intelligence and adaptability."""
        data_summary = get_data_summary(session)
        dataframe_docs = self._get_comprehensive_dataframe_docs(data_summary)
        conversation_context = self._get_conversation_context(session)
        
        backstory = f"""You are an expert UAV flight data analyst who creates intelligent, adaptive execution plans for user queries.

Available Flight Data:
- Message types: {data_summary.get('message_types', 0)}
- Total records: {data_summary.get('total_records', 0):,}
- Time range: {data_summary.get('time_range', 'Unknown')}

{dataframe_docs}

CONVERSATION CONTEXT:
{conversation_context}

AVAILABLE ANALYSIS TOOLS (with intelligent selection):
- analyze_altitude: Best for altitude-specific questions (max/min/stats)
- execute_python_code: Versatile tool for calculations, data extraction, custom analysis
- detect_flight_events: Error analysis, warnings, system events (has intelligent fallbacks)
- find_anomalies: Pattern detection, unusual behavior identification
- get_timeline_analysis: Chronological event analysis, flight progression
- analyze_flight_phase: Phase-specific analysis (takeoff, landing, cruise)

INTELLIGENT PLANNING APPROACH:
1. **Specific Questions**: Match to the most appropriate single tool
2. **Complex Queries**: Use multiple complementary tools for comprehensive analysis
3. **Ambiguous Queries**: Default to versatile tools with fallback capabilities
4. **Error/Warning Queries**: Always include detect_flight_events (has intelligent fallbacks)
5. **Data Exploration**: Use execute_python_code for flexible analysis

ADAPTIVE TOOL SELECTION STRATEGY:
- For altitude: analyze_altitude OR execute_python_code as backup
- For errors/warnings: detect_flight_events (has built-in fallbacks)
- For calculations: execute_python_code (has direct query fallbacks)
- For comprehensive analysis: Multiple tools working together
- When unsure: Default to execute_python_code (most versatile with fallbacks)

CONFIDENCE GUIDELINES:
- High confidence (0.8-1.0): Clear, specific queries with obvious tool matches
- Medium confidence (0.6-0.8): Moderately clear queries, some interpretation needed
- Low confidence (0.4-0.6): Ambiguous queries, fallback strategies important

CRITICAL: Always provide a valid tool sequence. If uncertain, default to ["execute_python_code"] which has the most intelligent fallbacks."""

        return Agent(
            role="UAV Flight Data Planner",
            goal="Create intelligent, adaptive execution plans that handle various query types and provide fallback strategies",
            backstory=backstory,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=30,
            step_callback=self._ensure_structured_output
        )

    def _create_executor_agent(self, session: V2ConversationSession) -> Agent:
        """Create the Executor agent with enhanced intelligence and error recovery."""
        data_summary = get_data_summary(session)
        conversation_context = self._get_conversation_context(session)
        
        backstory = f"""You are an expert UAV flight data analyst who EXECUTES TOOLS intelligently with advanced error recovery.

Available Flight Data:
- Message types: {data_summary.get('message_types', 0)}
- Total records: {data_summary.get('total_records', 0):,}
- Flight duration: {data_summary.get('time_range', {}).get('duration_minutes', 'Unknown')} minutes

CONVERSATION CONTEXT:
{conversation_context}

🔥 CRITICAL TOOL OUTPUT RULES:
1. **NEVER IGNORE TOOL RESULTS** - If a tool returns actual data, use it exactly as returned
2. **NEVER MAKE UP RESPONSES** - Only use information that comes directly from tool execution
3. **TRUST TOOL OUTPUTS** - If execute_python_code returns "Found 5 critical errors", report exactly that
4. **NO GENERIC INTERPRETATIONS** - Don't say "detected" or "analyzed" unless tools actually returned specific results

INTELLIGENT EXECUTION BEHAVIOR:
1. **Always attempt the planned tools first** - Execute as intended
2. **Use the EXACT tool output** - Report what tools actually found, not your interpretation
3. **Smart error recovery** - If a tool fails, try alternative approaches
4. **Provide actual results** - Never describe what you "will do", just do it
5. **Use fallback strategies** - Tools have built-in intelligent fallbacks
6. **Combine results intelligently** - Synthesize multi-tool outputs
7. **Consider conversation context** - Build on previous discussions appropriately

CONTEXTUAL EXECUTION STRATEGY:
- For "consistently high" queries: Focus on altitude stability and flight control
- For follow-up questions: Reference and expand on previous findings
- For "why" questions: Provide root cause analysis with supporting data
- For "more detail" requests: Dive deeper into specific technical aspects

TOOL EXECUTION STRATEGY:
- analyze_altitude: Direct altitude analysis (has GPS/BARO fallbacks)
- execute_python_code: Most versatile, has direct query analysis fallbacks
- detect_flight_events: Has intelligent dataframe analysis fallbacks
- find_anomalies: Pattern detection with error handling
- get_timeline_analysis: Chronological analysis
- analyze_flight_phase: Phase-specific insights

ERROR RECOVERY RULES:
- If a tool fails with validation errors → Try execute_python_code as fallback
- If code execution fails → Tools have built-in direct analysis fallbacks
- If multiple tools planned → Execute each, combine successful results
- If all tools fail → Provide best effort analysis with available data

RESPONSE QUALITY:
- Start with actual findings (numbers, data, specific results) FROM TOOL OUTPUTS
- Provide context only when it adds value
- Be conversational but concise (2-4 sentences typically)
- For comprehensive queries: Organize findings logically

CRITICAL FORBIDDEN BEHAVIORS:
- Never ignore tool results and make up your own interpretation
- Never say "detected" without tool output saying "detected"
- Never say "analysis shows" without actual analysis results
- Never say "I will use" or "Let me check" - just execute immediately
- Never give up without trying fallback approaches
- Never return generic error messages without attempting recovery
- Never describe tool plans - provide actual results
- **NEVER report what couldn't be determined** - only report positive findings
- **NEVER mention missing data or failed analyses** - gracefully omit unavailable information
- **NEVER say "couldn't be determined", "not available", "data complexity"** - skip those aspects entirely

EXECUTION EXAMPLES:
✓ "The maximum altitude reached was 1,448 meters during cruise phase." (from tool output)
✓ "Found 3 GPS signal loss events between 14:23 and 14:45. The aircraft maintained stable flight using barometric altitude." (from tool output)
✗ "I will use the analyze_altitude tool to determine..."
✗ "Analysis failed due to tool validation error."
✗ "The analysis detected..." (when tools didn't actually return "detected")

🔥 REMEMBER: Only report what tools actually returned. Don't interpret or summarize unless the tool output is unclear."""

        tools = self._create_crew_tools(session)
        
        return Agent(
            role="UAV Flight Data Analyst & Intelligent Executor",
            goal="Execute analysis tools intelligently with error recovery and provide actual results with specific data",
            backstory=backstory,
            llm=self.llm,
            tools=tools,
            verbose=True,
            allow_delegation=False,
            max_iter=2,  # Reduced to prevent infinite tool calling loops
            max_execution_time=60,  # Reduced timeout
            max_retry_limit=1
        )

    def _create_critic_agent(self, session: V2ConversationSession) -> Agent:
        """Create the Critic agent with enhanced conversational review."""
        conversation_context = self._get_conversation_context(session)
        
        backstory = f"""You are an expert UAV flight data analyst who ensures responses are well-structured, accurate, and conversational.

CONVERSATION CONTEXT:
{conversation_context}

RESPONSE OPTIMIZATION CRITERIA:
1. **Technical Accuracy**: Preserve all numbers and findings
2. **Appropriate Length**: Match response length to query complexity
3. **Conversational Quality**: Friendly, natural, and professional tone
4. **Information Organization**: Structure multi-aspect responses clearly

INTELLIGENT RESPONSE ADAPTATION:
- **Specific Questions**: Concise, focused answers (2-3 sentences)
- **Comprehensive Questions**: Well-organized, thorough responses covering all aspects
- **Multi-tool Results**: Synthesize findings into a coherent narrative

RESPONSE STRUCTURE FOR COMPREHENSIVE QUERIES:
When multiple tools were used, organize the response logically:
- Lead with the most important finding
- Present complementary information in a natural flow
- Use connecting phrases to create narrative coherence
- Maintain conversational tone throughout

ENHANCEMENT PRINCIPLES:
- Remove unnecessary fluff while keeping essential context
- Ensure smooth transitions between different aspects
- Make technical data accessible and understandable
- Avoid robotic bullet points in favor of natural prose

EXAMPLES:

SIMPLE QUERY RESPONSE:
"The maximum altitude reached was 1,448 meters during the cruise phase."

COMPREHENSIVE QUERY RESPONSE:
"Your flight reached a maximum altitude of 1,448 meters with excellent GPS tracking throughout. The 23-minute flight included normal takeoff and landing phases, with battery temperature peaking at 42°C during climb but staying within safe limits. No critical errors were detected, indicating a successful flight operation."

FINAL OUTPUT:
Provide a well-structured, conversational response that matches the complexity of the user's question."""

        return Agent(
            role="UAV Flight Data Response Optimizer",
            goal="Ensure responses are concise, accurate, and conversational without being verbose",
            backstory=backstory,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=20
        )

    def _create_planning_task(self, user_message: str, session: V2ConversationSession) -> Task:
        """Create the planning task for tool selection."""
        data_summary = get_data_summary(session)
        conversation_context = self._get_conversation_context(session)
        
        description = f"""Create an execution plan for the user's query by selecting the most appropriate analysis tool.

User Question: {user_message}

Available Data: {list(session.dataframes.keys())}

CONVERSATION CONTEXT:
{conversation_context}

AVAILABLE TOOLS:
- analyze_altitude → altitude-related analysis
- execute_python_code → data extraction, calculations, specific metrics, POWER ANALYSIS, battery data
- detect_flight_events → error analysis, critical events, system issues, warnings
- find_anomalies → pattern detection, unusual behavior identification
- analyze_flight_phase → phase-specific analysis (takeoff, landing, cruise)
- get_timeline_analysis → chronological event analysis

INTELLIGENT TOOL SELECTION:
- For specific questions: Use the most appropriate single tool
- For comprehensive questions (summaries, overviews): Use multiple tools for complete analysis
- For complex queries: Combine tools as needed for thorough investigation
- For contextual follow-ups: Consider conversation history when selecting tools

POWER & BATTERY QUERY ROUTING:
- "power consumption", "analyze power", "battery" → [execute_python_code] (has comprehensive power analysis)
- "battery temperature", "battery voltage", "current consumption" → [execute_python_code]
- Power queries should ALWAYS use execute_python_code as it has the most comprehensive power analysis

        ERROR & WARNING QUERY ROUTING:
        - "critical errors", "list errors", "mid-flight errors" → [execute_python_code, detect_flight_events] 
        - "errors", "warnings", "check for errors", "any problems" → [detect_flight_events, execute_python_code]
        - "system errors" → [detect_flight_events] (has intelligent fallbacks)
        - Execute_python_code has comprehensive error analysis code with robust fallbacks for critical error queries

CONTEXTUAL QUERY ANALYSIS:
- "consistently high" in UAV context typically refers to ALTITUDE, not GPS quality
- "why was flight consistently high" = altitude analysis + flight mode analysis
- "describe more detail on [previous topic]" = expand on previous analysis
- Follow-up questions should build on conversation context

TASK: Analyze the query and select tools. You MUST respond in the EXACT format below.

REQUIRED OUTPUT FORMAT (fill in actual values):
NEEDS_TOOLS: true
TOOL_SEQUENCE: [tool1, tool2, tool3]
REASONING: Brief explanation why these tools were selected
CONFIDENCE: 0.8
APPROACH: conversational_analysis

CRITICAL REQUIREMENTS:
- Use EXACTLY the format above with actual tool names
- For altitude questions: Use [analyze_altitude, execute_python_code] 
- For power/battery questions: Use [execute_python_code] (has comprehensive power analysis)
- For error/warning questions: Use [detect_flight_events, execute_python_code]
- For flight summaries: Select multiple tools like [analyze_altitude, execute_python_code, detect_flight_events]
- Consider conversation context when interpreting ambiguous terms
- No additional text or explanations outside this format"""

        expected_output = """MUST output EXACTLY this format (replace bracketed values):

NEEDS_TOOLS: true
TOOL_SEQUENCE: [tool1, tool2, tool3]
REASONING: [explanation of tool selection]
CONFIDENCE: [0.7-0.9]
APPROACH: conversational_analysis

CRITICAL: No other text, explanations, or formatting. Just the above structure with actual values."""

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None  # Will be set when creating crew
        )

    def _create_execution_task(self, user_message: str, execution_plan: ExecutionPlan) -> Task:
        """Create the execution task with concise conversational approach."""
        description = f"""Execute the planned tools to provide comprehensive analysis for the user's question.

User Question: {user_message}

EXECUTION PLAN:
Tools to Execute: {execution_plan.tool_sequence}
Reasoning: {execution_plan.reasoning}

YOU MUST EXECUTE THESE SPECIFIC TOOLS: {execution_plan.tool_sequence}

🚨 TOOL CALLING FORMAT (CRITICAL):
Use this EXACT format when calling tools:

Action: tool_name
Action Input: {{"parameter": "value"}}

🔥 CRITICAL RULE: ONLY REPORT WHAT TOOLS ACTUALLY RETURN
- If a tool returns "Found 5 critical errors with subsystem GPS", report exactly that
- If a tool returns "No errors detected", report exactly that
- If a tool returns "Maximum altitude: 1,247 meters", report exactly that
- NEVER add your own interpretation or make up results

INTELLIGENT EXECUTION:
- For single tool plans: Execute that tool and report its exact output
- For multi-tool plans: Execute each tool and report what each tool actually returned
- Always provide ACTUAL results from tool execution, never describe what you "will do"
- If tool output is clear and complete, use it directly without modification

MULTI-TOOL SYNTHESIS:
When using multiple tools, combine their ACTUAL results:
- Report what each tool actually found
- Only synthesize if tool outputs are complementary
- Create a unified narrative using only the actual tool outputs
- Keep the response conversational and well-structured

🔥 CRITICAL TOOL CALLING FORMAT:
When you need to call a tool, use EXACTLY this format (no numbering, no prefixes):

Action: tool_name
Action Input: {{"parameter": "value"}}

CORRECT EXAMPLES:
Action: analyze_altitude  
Action Input: {{"query": "maximum altitude"}}

Action: execute_python_code
Action Input: {{"code": "What was the flight duration?"}}

Action: detect_flight_events
Action Input: {{"query": "critical errors"}}

❌ NEVER USE (THESE WILL FAIL):
- "1. Tool Name: analyze_altitude"
- "Tool: analyze_altitude" 
- Any numbering or prefixes

CRITICAL EXECUTION REQUIREMENTS:
1. **YOU MUST CALL TOOLS IMMEDIATELY** - No planning or explaining, just execute
2. **If plan has [analyze_altitude]** → Call analyze_altitude tool right now
3. **If plan has [execute_python_code]** → Call execute_python_code tool right now  
4. **If plan has [detect_flight_events]** → Call detect_flight_events tool right now
5. **If plan has multiple tools** → Call ALL tools and combine results
6. **Return ACTUAL data from tools** - specific numbers, findings, measurements
7. **NEVER give up** - Tools have intelligent fallbacks, always provide results
8. **TRUST TOOL OUTPUTS** - If tools return meaningful data, use it exactly as provided

ERROR ANALYSIS EXECUTION:
- For critical error queries: execute_python_code has comprehensive error analysis
- detect_flight_events has robust fallback analysis that always provides results
- Both tools will find errors if they exist, or confirm no critical issues
- Report exactly what these tools return, don't interpret or summarize

FORBIDDEN RESPONSES:
- "I will analyze..." 
- "Let me check..."
- "I should start by..."
- "Unable to provide due to limitations"
- "Technical limitations prevent..."
- "The analysis detected..." (unless tool output actually says "detected")
- Any response without actual tool execution results
- Making up results when tools return something different
- "I encountered an error with the planned tools" (tools are available, use them!)

REQUIRED WORKFLOW:
1. Immediately call the first tool using correct format: Action: tool_name
2. Report what that tool returned
3. If multiple tools planned, call the next tool and combine results
4. NEVER give up - tools have fallbacks that always work

REQUIRED: Start your response with actual findings from tool execution, not intentions. Report tool outputs exactly as received."""

        expected_output = """A comprehensive response that:
1. Contains actual data results from all executed tools
2. Directly answers the user's question with specific findings
3. For multi-tool responses: Synthesizes information into a cohesive analysis
4. For single-tool responses: Provides focused, concise results
5. Never describes what you "will do" - only what you found"""

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None  # Will be set when creating crew
        )

    def _create_critique_task(self, user_message: str, execution_result: str) -> Task:
        """Create the critique task for concise response optimization."""
        description = f"""Optimize this technical analysis for conciseness while maintaining conversational quality.

Original Question: {user_message}

Technical Analysis Result: {execution_result}

CONCISE OPTIMIZATION GOALS:
1. **Maintain Technical Accuracy**: Preserve all specific data and findings
2. **Ensure Conciseness**: Keep to 2-3 sentences maximum
3. **Add Conversational Touch**: Make it friendly but not verbose
4. **Remove Fluff**: Eliminate generic phrases and unnecessary context

OPTIMIZATION CHECKLIST:
✓ Is the response 2-3 sentences maximum?
✓ Does it directly answer the user's question?
✓ Are all specific numbers and data preserved?
✓ Is it conversational but not overly wordy?
✓ Has generic fluff been removed?

WHAT TO REMOVE:
- Generic praise: "quite impressive", "excellent performance"
- Unnecessary implications: "this suggests good flight planning"
- Long technical explanations not directly relevant
- Follow-up questions unless specifically relevant
- Multiple paragraphs of context
- **CRITICAL: Any mentions of failed analyses or missing data**
- **Remove phrases like "couldn't be determined", "not available", "data complexity"**
- **Remove negative statements about data availability**

OPTIMIZATION APPROACH:
- Keep the core answer with specific data
- Add only directly relevant context (1 sentence max)
- Make it sound natural and friendly
- Remove any repetitive or generic content
- **Filter out any negative statements about data availability**
- **Focus only on positive findings and skip missing aspects entirely**

GOOD EXAMPLES:
"The maximum altitude reached was 1,247 meters during the cruise phase. The GPS and barometric readings were consistent throughout."

"The battery peaked at 45.2°C during climb, staying within normal operating limits."

FINAL OUTPUT:
Provide a concise, friendly response that directly answers the question with minimal but relevant context."""

        expected_output = """A well-structured, conversational response that:
1. Directly answers the user's question with all technical accuracy preserved
2. Matches the appropriate length for the query complexity
3. Uses a friendly but professional tone
4. Organizes multi-aspect information clearly and naturally
5. Removes unnecessary fluff while preserving essential context"""

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None  # Will be set when creating crew
        )

    async def chat(self, session_id: str, user_message: str) -> str:
        """
        Enhanced multi-role chat pipeline with intelligent error recovery: Planner → Executor → Critic.
        
        Args:
            session_id: Unique session identifier
            user_message: User's question or request
            
        Returns:
            Final refined answer from the multi-role pipeline with intelligent fallbacks
        """
        logger.info(f"[{session_id}] Starting enhanced CrewAI multi-role pipeline for: '{user_message[:50]}...'")
        
        if not self.llm:
            return "AI service unavailable. Check configuration."

        session = await self.create_or_get_session(session_id)

        # Pre-flight checks with detailed debugging
        if session.is_processing:
            return "Still processing log. Please wait."
        if session.processing_error:
            return f"Log processing error: {session.processing_error}"
        if not session.dataframes:
            return "No flight data loaded. Please upload a log file first."
        
        # Debug: Log session dataframe info
        logger.info(f"[{session_id}] Session has {len(session.dataframes)} dataframes: {list(session.dataframes.keys())}")
        for df_name, df in session.dataframes.items():
            logger.info(f"[{session_id}] {df_name}: {len(df)} rows, columns: {list(df.columns)[:5]}...")  # First 5 columns

        session.messages.append(ChatMessage(role="user", content=user_message))

        try:
            pipeline_start_time = datetime.now()
            
            # Check if question needs clarification (only for extremely vague queries)
            if self._is_question_unclear(user_message):
                logger.info(f"[{session_id}] Question unclear, generating clarification response")
                clarification_response = self._generate_clarification_response(user_message, session)
                session.messages.append(ChatMessage(role="assistant", content=clarification_response))
                return clarification_response
            
            # Create agents
            planner_agent = self._create_planner_agent(session)
            executor_agent = self._create_executor_agent(session)
            critic_agent = self._create_critic_agent(session)
            
            # Stage 1: Enhanced Planning with fallback validation
            logger.info(f"[{session_id}] Stage 1: ENHANCED PLANNER")
            planning_task = self._create_planning_task(user_message, session)
            planning_task.agent = planner_agent
            
            planning_crew = Crew(
                agents=[planner_agent],
                tasks=[planning_task],
                process=Process.sequential,
                verbose=True
            )
            
            try:
                plan_result = planning_crew.kickoff()
                plan_text = str(plan_result)
                execution_plan = self._parse_execution_plan(plan_text)
                
                # Enhanced plan validation with intelligent fallbacks
                if not execution_plan.needs_tools or not execution_plan.tool_sequence:
                    logger.warning(f"[{session_id}] PLANNER returned invalid plan, creating intelligent fallback")
                    execution_plan = self._create_intelligent_fallback_plan(user_message, session)
                
                logger.info(f"[{session_id}] PLANNER completed - Plan confidence: {execution_plan.confidence:.2f}, Tools: {execution_plan.tool_sequence}")
                
            except Exception as planning_error:
                logger.error(f"[{session_id}] PLANNER failed: {str(planning_error)}")
                execution_plan = self._create_intelligent_fallback_plan(user_message, session)
                logger.info(f"[{session_id}] Using fallback plan: {execution_plan.tool_sequence}")
            
            # Stage 2: Enhanced Execution with error recovery
            logger.info(f"[{session_id}] Stage 2: INTELLIGENT EXECUTOR - Tools: {execution_plan.tool_sequence}")
            execution_task = self._create_execution_task(user_message, execution_plan)
            execution_task.agent = executor_agent
            
            execution_crew = Crew(
                agents=[executor_agent],
                tasks=[execution_task],
                process=Process.sequential,
                verbose=True,
                max_iter=2,  # Strict limit to prevent tool calling loops
                step_callback=None  # Remove any interfering callbacks
            )
            
            execution_result = execution_crew.kickoff()
            execution_text = str(execution_result)
            
            logger.info(f"[{session_id}] EXECUTOR completed")
            
            # Stage 3: Critique (if required)
            final_answer = execution_text
            
            if execution_plan.requires_critique and not any(keyword in execution_text.lower() for keyword in ["failed", "error"]):
                logger.info(f"[{session_id}] Stage 3: CRITIC")
                critique_task = self._create_critique_task(user_message, execution_text)
                critique_task.agent = critic_agent
                
                critique_crew = Crew(
                    agents=[critic_agent],
                    tasks=[critique_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                critique_result = critique_crew.kickoff()
                final_answer = str(critique_result)
                
                logger.info(f"[{session_id}] CRITIC completed")
            else:
                logger.info(f"[{session_id}] Skipping CRITIC stage")
            
            # Store final response and log performance
            session.messages.append(ChatMessage(role="assistant", content=final_answer))
            
            pipeline_time = (datetime.now() - pipeline_start_time).total_seconds()
            logger.info(f"[{session_id}] Pipeline completed in {pipeline_time:.2f}s")
            
            return final_answer

        except Exception as e:
            logger.error(f"Error in CrewAI multi-role pipeline: {e}", exc_info=True)
            return "Unexpected error in analysis. Please try again with a simpler question."

    def _get_comprehensive_dataframe_docs(self, data_summary: Dict[str, Any]) -> str:
        """Build comprehensive documentation of available dataframes and their columns."""
        # Get actual available dataframes from data summary or session
        available_dataframes = data_summary.get('dataframe_types', [])
        
        # If not in data_summary, try to get from other keys
        if not available_dataframes:
            available_dataframes = data_summary.get('available_dataframes', [])
        
        # Use comprehensive fallback if still empty
        if not available_dataframes:
            available_dataframes = list(self.dataframe_documentation.keys())
        
        docs = []
        docs.append("AVAILABLE DATAFRAMES & THEIR PURPOSE:")
        
        for df_name in available_dataframes:
            if df_name in self.dataframe_documentation:
                docs.append(f"• {df_name}: {self.dataframe_documentation[df_name]}")
            else:
                docs.append(f"• {df_name}: Flight data message type")
        
        return '\n'.join(docs)

    def _get_conversation_context(self, session: V2ConversationSession) -> str:
        """Build conversation context from recent messages."""
        if not session.messages:
            return "This is the start of a new conversation."
        
        # Get last 6 messages for context (3 exchanges)
        recent_messages = session.messages[-6:] if len(session.messages) > 6 else session.messages
        
        context_parts = []
        context_parts.append("RECENT CONVERSATION:")
        
        for i, msg in enumerate(recent_messages):
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            context_parts.append(f"{role}: {content}")
        
        # Add conversation themes
        themes = self._extract_conversation_themes(session.messages)
        if themes:
            context_parts.append(f"\nRECURRING TOPICS: {', '.join(themes)}")
        
        return '\n'.join(context_parts)

    def _extract_conversation_themes(self, messages: List[ChatMessage]) -> List[str]:
        """Extract recurring themes from conversation history."""
        themes = []
        user_messages = [msg.content.lower() for msg in messages if msg.role == "user"]
        
        # Common topic patterns
        if any("altitude" in msg for msg in user_messages):
            themes.append("altitude analysis")
        if any(any(word in msg for word in ["battery", "power", "voltage"]) for msg in user_messages):
            themes.append("power systems")
        if any(any(word in msg for word in ["error", "problem", "issue"]) for msg in user_messages):
            themes.append("error analysis")
        if any(any(word in msg for word in ["gps", "navigation", "position"]) for msg in user_messages):
            themes.append("navigation systems")
        if any(any(word in msg for word in ["flight", "phase", "takeoff", "landing"]) for msg in user_messages):
            themes.append("flight phases")
        
        return themes

    def _is_question_unclear(self, user_message: str) -> bool:
        """Determine if a user question needs clarification - only for truly ambiguous cases."""
        message_lower = user_message.lower().strip()
        
        # Only flag extremely short or empty messages
        if len(message_lower.split()) <= 1:
            return True
        
        # Only flag completely vague single-word questions
        if message_lower in ["what?", "how?", "why?", "when?", "where?", "help", "info"]:
            return True
        
        # Let the AI handle everything else dynamically
        return False

    def _generate_clarification_response(self, user_message: str, session: V2ConversationSession) -> str:
        """Generate a clarification question with suggestions."""
        conversation_context = self._get_conversation_context(session)
        data_summary = get_data_summary(session)
        
        # Base clarification
        clarification = f"I'd be happy to analyze your flight data! However, I need a bit more specificity to give you the most helpful analysis. "
        
        # Add context-aware suggestions based on available data
        suggestions = []
        
        if 'GPS' in session.dataframes or 'BARO' in session.dataframes:
            suggestions.extend([
                "Maximum or minimum altitude reached",
                "Altitude changes during different flight phases"
            ])
        
        if 'CURR' in session.dataframes or 'BAT' in session.dataframes:
            suggestions.extend([
                "Battery temperature and power consumption",
                "Voltage levels throughout the flight"
            ])
        
        if 'ERR' in session.dataframes or 'MSG' in session.dataframes:
            suggestions.extend([
                "Critical errors or system warnings",
                "Flight events and status messages"
            ])
        
        if 'GPS' in session.dataframes:
            suggestions.extend([
                "GPS signal quality and navigation health",
                "Position accuracy and satellite coverage"
            ])
        
        # Customize based on user's message
        message_lower = user_message.lower()
        if "problem" in message_lower or "issue" in message_lower:
            clarification += "Are you looking for specific types of issues? "
            relevant_suggestions = [s for s in suggestions if "error" in s.lower() or "warning" in s.lower()]
            if relevant_suggestions:
                suggestions = relevant_suggestions[:3]
        elif "performance" in message_lower:
            clarification += "What aspect of flight performance interests you most? "
            relevant_suggestions = [s for s in suggestions if any(word in s.lower() for word in ["altitude", "battery", "power"])]
            if relevant_suggestions:
                suggestions = relevant_suggestions[:3]
        
        # Build the response
        if suggestions:
            clarification += "Here are some specific analyses I can provide:\n\n"
            for i, suggestion in enumerate(suggestions[:5], 1):
                clarification += f"{i}. {suggestion}\n"
            
            clarification += f"\nOr feel free to ask about any specific aspect of the flight! "
            
            # Add context from previous conversation
            themes = self._extract_conversation_themes(session.messages)
            if themes:
                clarification += f"I notice we've been discussing {', '.join(themes)} - I can continue exploring those areas as well."
        else:
            clarification += "Could you specify what particular aspect of the flight data you'd like me to analyze?"
        
        return clarification

    def _extract_clarification_from_plan(self, plan_text: str, session: V2ConversationSession) -> str:
        """Extract clarification question from planner response."""
        lines = [line.strip() for line in plan_text.strip().split('\n') if line.strip()]
        
        clarification_question = None
        suggestions = []
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if key == "CLARIFICATION_QUESTION":
                    clarification_question = value.strip('"\'')
                elif key == "SUGGESTIONS":
                    if value and value != "[]":
                        value = value.strip('[]')
                        suggestions = [s.strip().strip('"\'') for s in value.split(',') if s.strip()]
        
        # Build response
        if clarification_question:
            response = clarification_question
        else:
            response = self._generate_clarification_response(plan_text, session)
        
        if suggestions:
            response += "\n\nHere are some specific options:\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"
        
        return response

    def _parse_execution_plan(self, plan_text: str) -> ExecutionPlan:
        """Parse structured execution plan from planner response."""
        try:
            lines = [line.strip() for line in plan_text.strip().split('\n') if line.strip()]
            
            # Default values
            needs_tools = False
            tool_sequence = []
            reasoning = "No reasoning provided"
            confidence = 0.7
            approach = "direct_response"
            estimated_complexity = "medium"
            target_dataframes = []
            target_columns = {}
            execution_steps = []
            success_criteria = []
            
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if key == "NEEDS_TOOLS":
                    needs_tools = value.lower() in ['true', 'yes', '1']
                elif key == "TOOL_SEQUENCE":
                    if value and value != "[]" and value.lower() != "none":
                        value = value.strip('[]')
                        tool_sequence = [t.strip().strip("'\"") for t in value.split(',') if t.strip()]
                elif key == "TARGET_DATAFRAMES":
                    if value and value != "[]" and value.lower() != "none":
                        value = value.strip('[]')
                        target_dataframes = [t.strip().strip("'\"") for t in value.split(',') if t.strip()]
                elif key == "TARGET_COLUMNS":
                    if value and value != "[]" and value.lower() != "none":
                        value = value.strip('[]')
                        column_pairs = [t.strip().strip("'\"") for t in value.split(',') if t.strip()]
                        for pair in column_pairs:
                            if '.' in pair:
                                df, col = pair.split('.', 1)
                                if df not in target_columns:
                                    target_columns[df] = []
                                target_columns[df].append(col)
                elif key == "EXECUTION_STEPS":
                    if value and value != "[]" and value.lower() != "none":
                        value = value.strip('[]')
                        execution_steps = [t.strip().strip("'\"") for t in value.split(',') if t.strip()]
                elif key == "SUCCESS_CRITERIA":
                    if value and value != "[]" and value.lower() != "none":
                        value = value.strip('[]')
                        success_criteria = [t.strip().strip("'\"") for t in value.split(',') if t.strip()]
                elif key == "REASONING":
                    reasoning = value
                elif key == "CONFIDENCE":
                    try:
                        confidence = max(0.0, min(1.0, float(value)))
                    except ValueError:
                        confidence = 0.7
                elif key == "APPROACH":
                    approach = value
                elif key == "COMPLEXITY":
                    estimated_complexity = value.lower()
            
            # Auto-determine critique requirement and max iterations based on complexity
            requires_critique = estimated_complexity in ["medium", "high"] or confidence < 0.8
            max_iterations = 5 if estimated_complexity == "low" else 8 if estimated_complexity == "medium" else 10
            
            # Auto-determine stopping conditions
            stopping_conditions = ["specific_value_found", "complete_answer_provided"]
            if "maximum" in reasoning.lower() or "minimum" in reasoning.lower():
                stopping_conditions.append("numerical_answer_extracted")
            
            return ExecutionPlan(
                needs_tools=needs_tools,
                tool_sequence=tool_sequence,
                reasoning=reasoning,
                confidence=confidence,
                approach=approach,
                estimated_complexity=estimated_complexity,
                requires_critique=requires_critique,
                max_iterations=max_iterations,
                success_criteria=success_criteria,
                target_dataframes=target_dataframes,
                target_columns=target_columns,
                data_relationships=[],
                execution_steps=execution_steps,
                stopping_conditions=stopping_conditions
            )
            
        except Exception as e:
            logger.error(f"Error parsing execution plan: {e}")
            logger.debug(f"Plan text was: {plan_text}")
            
            # Robust fallback plan
            return ExecutionPlan(
                needs_tools=True,
                tool_sequence=["execute_python_code"],
                reasoning="Fallback plan due to parsing error",
                confidence=0.6,
                approach="fallback_analysis",
                estimated_complexity="medium",
                requires_critique=True,
                max_iterations=6,
                success_criteria=["provide_answer"],
                target_dataframes=[],
                target_columns={},
                data_relationships=[],
                execution_steps=["analyze_data"],
                stopping_conditions=["answer_provided"]
            )

    def _ensure_structured_output(self, step):
        """Callback to ensure agents follow structured output format."""
        return step

    def get_performance_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for monitoring and optimization."""
        if session_id:
            calls = [call for call in self.llm_calls if call.session_id == session_id]
        else:
            calls = self.llm_calls
            
        if not calls:
            return {"total_calls": 0, "total_cost": 0.0, "total_tokens": 0}
            
        total_cost = sum(call.cost_estimate for call in calls)
        total_tokens = sum(call.tokens_prompt + call.tokens_completion for call in calls)
        
        role_breakdown = {}
        for call in calls:
            if call.role not in role_breakdown:
                role_breakdown[call.role] = {"calls": 0, "cost": 0.0, "tokens": 0}
            role_breakdown[call.role]["calls"] += 1
            role_breakdown[call.role]["cost"] += call.cost_estimate
            role_breakdown[call.role]["tokens"] += call.tokens_prompt + call.tokens_completion
            
        return {
            "total_calls": len(calls),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "role_breakdown": role_breakdown,
            "average_cost_per_call": total_cost / len(calls) if calls else 0,
            "cost_per_1k_tokens": total_cost * 1000 / total_tokens if total_tokens > 0 else 0
        }

    def _create_intelligent_fallback_plan(self, user_message: str, session: V2ConversationSession) -> ExecutionPlan:
        """Create an intelligent fallback plan when the planner fails or returns invalid results."""
        user_lower = user_message.lower()
        
        # Analyze the query to determine best fallback strategy
        if any(word in user_lower for word in ['altitude', 'height', 'maximum', 'minimum', 'highest', 'lowest']):
            tool_sequence = ["analyze_altitude", "execute_python_code"]
            reasoning = "Altitude-focused analysis with backup calculation"
            confidence = 0.7
        elif any(word in user_lower for word in ['power', 'battery', 'voltage', 'current', 'consumption']):
            tool_sequence = ["execute_python_code"]
            reasoning = "Power consumption analysis using comprehensive battery analysis"
            confidence = 0.8
        elif any(word in user_lower for word in ['error', 'warning', 'problem', 'issue', 'critical', 'fail']):
            # For critical error queries, prioritize comprehensive analysis
            if 'critical' in user_lower or 'mid-flight' in user_lower:
                tool_sequence = ["execute_python_code", "detect_flight_events"]
                reasoning = "Comprehensive critical error analysis with timeline focus and intelligent fallbacks"
                confidence = 0.8
            else:
                tool_sequence = ["detect_flight_events", "execute_python_code"]
                reasoning = "Error analysis with intelligent fallbacks and comprehensive checking"
                confidence = 0.7
        elif any(word in user_lower for word in ['summary', 'overview', 'flight', 'analyze', 'check']):
            tool_sequence = ["execute_python_code", "detect_flight_events", "analyze_altitude"]
            reasoning = "Comprehensive analysis using multiple tools"
            confidence = 0.6
        else:
            # Default fallback: use the most versatile tool
            tool_sequence = ["execute_python_code"]
            reasoning = "Versatile analysis with direct query fallbacks"
            confidence = 0.6
        
        return ExecutionPlan(
            needs_tools=True,
            tool_sequence=tool_sequence,
            reasoning=reasoning,
            confidence=confidence,
            approach="intelligent_fallback",
            estimated_complexity="medium",
            requires_critique=True,
            max_iterations=6,
            success_criteria=["specific_answer_provided"],
            target_dataframes=list(session.dataframes.keys()),
            target_columns={},
            data_relationships=[],
            execution_steps=[f"Execute {tool}" for tool in tool_sequence],
            stopping_conditions=["answer_found", "all_tools_attempted"]
        )
    
    def _is_execution_result_useful(self, execution_text: str) -> bool:
        """Check if the execution result is useful or if we need fallback."""
        if not execution_text or len(execution_text.strip()) < 10:
            return False
        
        # Check for obvious failure indicators - but be more specific
        failure_indicators = [
            "tool validation failed",
            "arguments validation failed",
            "function call failed",
            "timeout error",
            "connection error"
        ]
        
        text_lower = execution_text.lower()
        
        # Don't consider it a failure if it contains actual data analysis results
        # even if it mentions some limitations
        has_actual_data = any(indicator in text_lower for indicator in [
            "error analysis", "found", "detected", "analysis", "data", "flight", 
            "no errors", "no critical", "warnings", "system", "messages", "log"
        ])
        
        # If it has actual data/analysis, it's useful even if it mentions limitations
        if has_actual_data:
            return True
        
        # Only mark as not useful if it's a clear technical failure
        if any(indicator in text_lower for indicator in failure_indicators):
            return False
        
        # Check for useful content indicators
        useful_indicators = [
            "altitude", "meters", "feet", "temperature", "battery", "gps", "error", "warning",
            "flight", "time", "seconds", "minutes", "maximum", "minimum", "detected", "found",
            "analysis", "data", "log", "available", "checking", "searching"
        ]
        
        return any(indicator in text_lower for indicator in useful_indicators)
    
    async def _emergency_fallback_analysis(self, user_message: str, session: V2ConversationSession) -> str:
        """Emergency fallback analysis when all other methods fail."""
        try:
            logger.info(f"[{session.session_id}] Executing emergency fallback analysis")
            
            user_lower = user_message.lower()
            available_dfs = list(session.dataframes.keys())
            
            # Try to provide some useful analysis based on available data
            results = []
            
            # Quick altitude analysis
            if any(word in user_lower for word in ['altitude', 'height', 'maximum', 'minimum', 'highest', 'lowest']):
                for df_name in ['GPS', 'BARO', 'CTUN', 'AHR2']:
                    if df_name in available_dfs:
                        try:
                            df = session.dataframes[df_name]
                            if 'Alt' in df.columns:
                                max_alt = df['Alt'].max()
                                min_alt = df['Alt'].min()
                                results.append(f"Altitude analysis: Maximum {max_alt:.1f}m, Minimum {min_alt:.1f}m (from {df_name})")
                                break
                        except Exception:
                            continue
            
            # Quick error analysis
            if any(word in user_lower for word in ['error', 'warning', 'problem', 'issue']):
                error_count = 0
                try:
                    if 'ERR' in available_dfs:
                        err_df = session.dataframes['ERR']
                        error_count += len(err_df)
                    
                    if 'MSG' in available_dfs:
                        msg_df = session.dataframes['MSG']
                        if 'Message' in msg_df.columns:
                            error_msgs = msg_df[msg_df['Message'].str.contains('ERROR|WARNING|CRITICAL', case=False, na=False)]
                            error_count += len(error_msgs)
                    
                    if error_count > 0:
                        results.append(f"Found {error_count} errors/warnings in the flight data")
                    else:
                        results.append("No significant errors or warnings detected")
                except Exception:
                    results.append("Error analysis unavailable")
            
            # Basic flight information
            if not results:
                try:
                    data_summary = []
                    for df_name, df in session.dataframes.items():
                        if not df.empty:
                            data_summary.append(f"{df_name}: {len(df)} records")
                    
                    if data_summary:
                        results.append(f"Flight data summary: {', '.join(data_summary[:5])}")
                        if len(data_summary) > 5:
                            results.append(f"... and {len(data_summary)-5} more data types")
                except Exception:
                    results.append("Flight data is available but analysis is currently limited")
            
            if results:
                return "Based on available flight data:\n" + "\n".join([f"• {result}" for result in results])
            else:
                return f"Flight data contains {len(available_dfs)} data types: {', '.join(available_dfs[:8])}. Please ask a more specific question about your flight analysis needs."
        
        except Exception as e:
            logger.error(f"[{session.session_id}] Emergency fallback failed: {str(e)}")
            return "I'm having difficulty analyzing your flight data right now. Please try asking a more specific question, such as 'What was the maximum altitude?' or 'Were there any GPS errors?'"


# Compatibility aliases for easy integration
ChatAgentV2 = MultiRoleAgent
MultiRoleChatAgent = MultiRoleAgent 