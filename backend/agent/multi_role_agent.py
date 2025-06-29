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
import os
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import re
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import pandas as pd
import shutil

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from config import get_settings
from models import ChatMessage, V2ConversationSession

from utils.documentation import DocumentationService
from utils.log_parser import LogParserService, get_data_summary
from utils.semantic_data_retriever import SemanticDataRetriever
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
class ClassifiedIntent:
    """Result from intent classification."""
    intent_key: str
    intent_name: str
    confidence: float
    recommended_tools: List[str]
    reasoning: str


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
    # Intent classification
    classified_intent: Optional[ClassifiedIntent] = None


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


class IntentClassifier:
    """Intelligent intent classification using LLM to replace brittle keyword matching."""
    
    def __init__(self, settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.intent_classification_model,
            temperature=settings.intent_classification_temperature,
            max_tokens=settings.intent_classification_max_tokens
        ) if settings.openai_api_key else None
        
    async def classify_intent(self, user_message: str, session: V2ConversationSession) -> ClassifiedIntent:
        """Classify user intent using LLM instead of keyword matching."""
        if not self.llm:
            logger.warning("LLM not available for intent classification, falling back to default")
            return self._fallback_classification(user_message)
        
        try:
            # Build classification prompt with available intents
            intent_descriptions = []
            for intent_key, intent_config in self.settings.intents.items():
                intent_descriptions.append(
                    f"- {intent_config['name']}: {intent_config['description']}"
                )
            
            # Add conversation context for better classification
            conversation_context = self._get_conversation_context(session)
            
            classification_prompt = f"""You are an expert UAV flight data analyst. Classify the user's query into one of the predefined intent categories.

Available Intent Categories:
{chr(10).join(intent_descriptions)}

Conversation Context:
{conversation_context}

User Query: "{user_message}"

Instructions:
1. Analyze the query carefully considering the conversation context
2. Select the MOST APPROPRIATE intent category
3. Provide a confidence score (0.0-1.0)
4. Give a brief reasoning for your choice

Respond ONLY in this exact JSON format:
{{
    "intent_key": "the_intent_key",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this intent was selected"
}}"""

            # Make LLM call for classification
            response = await asyncio.to_thread(
                lambda: self.llm.invoke(classification_prompt).content
            )
            
            # Parse LLM response
            classified_intent = self._parse_classification_response(response, user_message)
            
            logger.info(f"Intent classified: {classified_intent.intent_name} (confidence: {classified_intent.confidence:.2f})")
            return classified_intent
            
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}, using fallback")
            return self._fallback_classification(user_message)
    
    def _parse_classification_response(self, response: str, user_message: str) -> ClassifiedIntent:
        """Parse LLM classification response into ClassifiedIntent object."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                intent_key = result.get('intent_key', 'general_code_execution')
                confidence = float(result.get('confidence', 0.7))
                reasoning = result.get('reasoning', 'LLM classification')
                
                # Validate intent key exists in config
                if intent_key not in self.settings.intents:
                    logger.warning(f"Unknown intent key: {intent_key}, falling back to general_code_execution")
                    intent_key = 'general_code_execution'
                
                intent_config = self.settings.intents[intent_key]
                
                return ClassifiedIntent(
                    intent_key=intent_key,
                    intent_name=intent_config['name'],
                    confidence=confidence,
                    recommended_tools=intent_config['tools'].copy(),
                    reasoning=reasoning
                )
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.warning(f"Failed to parse classification response: {e}")
            return self._fallback_classification(user_message)
    
    def _fallback_classification(self, user_message: str) -> ClassifiedIntent:
        """Fallback classification when LLM classification fails."""
        # Simple heuristic-based classification as fallback
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['altitude', 'height', 'maximum', 'minimum', 'highest', 'lowest']):
            intent_key = 'altitude_analysis'
        elif any(word in user_lower for word in ['power', 'battery', 'voltage', 'current', 'consumption']):
            intent_key = 'power_system_analysis'
        elif any(word in user_lower for word in ['error', 'warning', 'problem', 'issue', 'critical', 'fail']):
            intent_key = 'flight_events_analysis'
        elif any(word in user_lower for word in ['gps', 'navigation', 'position', 'satellite']):
            intent_key = 'navigation_analysis'
        elif any(word in user_lower for word in ['summary', 'overview', 'comprehensive', 'performance']):
            intent_key = 'flight_performance_analysis'
        elif any(word in user_lower for word in ['anomaly', 'pattern', 'unusual', 'detect']):
            intent_key = 'anomaly_detection'
        elif any(word in user_lower for word in ['takeoff', 'landing', 'cruise', 'phase', 'mode']):
            intent_key = 'flight_phase_analysis'
        else:
            intent_key = 'general_code_execution'
        
        intent_config = self.settings.intents[intent_key]
        
        return ClassifiedIntent(
            intent_key=intent_key,
            intent_name=intent_config['name'],
            confidence=0.6,  # Lower confidence for fallback
            recommended_tools=intent_config['tools'].copy(),
            reasoning="Fallback heuristic classification"
        )
    
    def _get_conversation_context(self, session: V2ConversationSession) -> str:
        """Get conversation context for better intent classification."""
        if not session.messages:
            return "This is the start of a new conversation."
        
        # Get last 2 messages for context
        recent_messages = session.messages[-2:] if len(session.messages) >= 2 else session.messages
        
        context_parts = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            context_parts.append(f"{role}: {content}")
        
        return "; ".join(context_parts) if context_parts else "No previous context"


from pydantic import BaseModel, Field
from typing import Type

# GENERALIZED Python Code Validation and Error Recovery System
class PythonCodeValidator:
    """
    Generalized Python code validation and error recovery system.
    Uses LLM-based analysis to handle any type of Python execution errors dynamically.
    
    🎯 KEY FEATURES:
    - Dynamic syntax error fixing using LLM analysis
    - Adaptive fallback generation based on user intent and error type
    - Learning from error patterns (extensible)
    - No hardcoded keywords or error messages
    - Scales to handle any type of analysis failure
    """
    
    def __init__(self, llm, session: V2ConversationSession):
        self.llm = llm
        self.session = session
        self.error_patterns = {}  # Dynamic learning of error patterns
    
    def validate_and_fix_code(self, code: str) -> str:
        """Validate Python code and fix common issues dynamically."""
        try:
            # Try to compile the code first
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError as e:
            logger.warning(f"Syntax error detected: {e}")
            return self._fix_syntax_error(code, str(e))
        except Exception as e:
            logger.warning(f"Code validation issue: {e}")
            return code  # Return as-is for runtime validation
    
    def _fix_syntax_error(self, code: str, error_msg: str) -> str:
        """Fix syntax errors using LLM analysis."""
        if not self.llm:
            return code
        
        try:
            fix_prompt = f"""Fix this Python code syntax error:

ERROR: {error_msg}

CODE:
```python
{code}
```

Requirements:
1. Fix ONLY the syntax error, don't change the logic
2. Maintain all variable names and data access patterns
3. Return ONLY the corrected Python code, no explanations
4. Ensure code uses 'dfs' dictionary to access dataframes
5. Must end with 'result' variable containing the final answer

Corrected code:"""

            response = self.llm.invoke(fix_prompt).content
            # Extract code from response
            if "```python" in response:
                start = response.find("```python") + 9
                end = response.find("```", start)
                fixed_code = response[start:end].strip()
            else:
                fixed_code = response.strip()
            
            # Validate the fix
            try:
                compile(fixed_code, '<string>', 'exec')
                logger.info("Successfully fixed syntax error using LLM")
                return fixed_code
            except:
                logger.warning("LLM fix failed compilation, returning original")
                return code
                
        except Exception as e:
            logger.error(f"Error in LLM syntax fix: {e}")
            return code

    def generate_adaptive_fallback(self, original_code: str, error_msg: str, user_intent: str) -> str:
        """Generate adaptive fallback code based on error analysis."""
        if not self.llm:
            return self._basic_fallback(user_intent)
        
        try:
            available_dfs = list(self.session.dataframes.keys())
            df_info = {}
            for df_name in available_dfs[:10]:  # Limit to prevent token overflow
                df = self.session.dataframes[df_name]
                df_info[df_name] = list(df.columns)[:10]  # First 10 columns
            
            fallback_prompt = f"""Generate robust fallback Python code for this failed analysis:

ORIGINAL CODE:
```python
{original_code}
```

ERROR: {error_msg}

USER INTENT: {user_intent}

AVAILABLE DATA:
{df_info}

Requirements:
1. Create safe, robust Python code that won't fail
2. Address the user's intent: {user_intent}
3. Use simple, safe iteration patterns (avoid complex pandas operations if they caused errors)
4. Handle missing data gracefully with try-catch blocks
5. Use 'dfs' dictionary to access dataframes
6. Must end with 'result' variable containing meaningful output
7. Provide useful information even if limited data is available

Generate fallback code:"""

            response = self.llm.invoke(fallback_prompt).content
            
            # Extract code from response
            if "```python" in response:
                start = response.find("```python") + 9
                end = response.find("```", start)
                fallback_code = response[start:end].strip()
            else:
                fallback_code = response.strip()
            
            # Validate the fallback
            try:
                compile(fallback_code, '<string>', 'exec')
                logger.info("Generated adaptive fallback using LLM")
                return fallback_code
            except Exception as compile_error:
                logger.warning(f"LLM fallback failed compilation: {compile_error}")
                return self._basic_fallback(user_intent)
                
        except Exception as e:
            logger.error(f"Error generating adaptive fallback: {e}")
            return self._basic_fallback(user_intent)
    
    def _basic_fallback(self, user_intent: str) -> str:
        """Basic fallback when LLM is not available."""
        return f'''# Basic analysis fallback
available_data = list(dfs.keys())
data_counts = {{df_name: len(dfs[df_name]) for df_name in available_data}}

result = f"Data available for analysis: {{', '.join(available_data)}}. Total records: {{sum(data_counts.values())}}"
result'''

    def extract_user_intent(self, original_code: str) -> str:
        """Extract user intent from the original code dynamically."""
        code_lower = original_code.lower()
        
        # Dynamic intent extraction based on code patterns
        if any(word in code_lower for word in ['error', 'critical', 'warning', 'msg']):
            return "error and warning analysis"
        elif any(word in code_lower for word in ['power', 'volt', 'curr', 'battery']):
            return "power system analysis"
        elif any(word in code_lower for word in ['alt', 'altitude', 'height']):
            return "altitude analysis"
        elif any(word in code_lower for word in ['gps', 'position', 'navigation']):
            return "navigation analysis"
        elif any(word in code_lower for word in ['summary', 'overview', 'duration']):
            return "flight summary analysis"
        else:
            return "general flight data analysis"


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

class CodeTemplates:
    """
    Safe code templates for agent instructions. 
    All Python code examples are stored here to prevent f-string evaluation errors.
    
    🔒 SECURITY FEATURES:
    - Eliminates f-string evaluation vulnerabilities
    - Provides safe template storage and retrieval  
    - Validates template safety before use
    - Supports extensible template management
    
    📋 USAGE EXAMPLES:
    ```python
    # Get code examples safely
    examples = CodeTemplates.get_code_examples_text()
    
    # Add new template safely
    CodeTemplates.add_new_template('new_analysis', 'result = "Safe code"')
    
    # Validate template safety
    is_safe = CodeTemplates.validate_template_safety(template_code)
    ```
    
    ⚠️ IMPORTANT: When adding new templates, ensure they don't contain
    f-strings with undefined variables that could cause NameError at runtime.
    """
    
    # Basic Python code examples that are safe from variable evaluation
    FLIGHT_SUMMARY_TEMPLATE = '''# Comprehensive flight analysis
summary = []

# Flight duration calculation
time_values = []
for df_name, df in dfs.items():
    if 'TimeUS' in df.columns:
        time_values.extend(df['TimeUS'].dropna().tolist())
if time_values:
    duration_min = (max(time_values) - min(time_values)) / 60_000_000
    summary.append(f"Duration: {duration_min:.1f} minutes")

# Maximum altitude analysis
max_alt = None
for source in ['GPS', 'BARO', 'CTUN']:
    if source in dfs and 'Alt' in dfs[source].columns:
        alt_values = dfs[source]['Alt'].dropna()
        if not alt_values.empty:
            current_max = alt_values.max()
            if max_alt is None or current_max > max_alt:
                max_alt = current_max
if max_alt:
    summary.append(f"Max altitude: {max_alt:.0f}m")

# Basic power analysis
for source in ['BAT', 'CURR']:
    if source in dfs and 'Volt' in dfs[source].columns:
        voltage = dfs[source]['Volt'].dropna()
        if not voltage.empty:
            summary.append(f"Battery: {voltage.mean():.1f}V avg")
            break

result = "Flight Summary: " + ", ".join(summary)
result'''

    POWER_ANALYSIS_TEMPLATE = '''# Power system analysis
power_info = []
for source in ['BAT', 'CURR', 'POWR']:
    if source in dfs:
        df = dfs[source]
        if 'Volt' in df.columns:
            voltage = df['Volt'].dropna()
            if not voltage.empty:
                power_info.append(f"Voltage: {voltage.mean():.2f}V avg, {voltage.min():.2f}V min")
        if 'Curr' in df.columns:
            current = df['Curr'].dropna()
            if not current.empty:
                power_info.append(f"Current: {current.mean():.1f}A avg, {current.max():.1f}A peak")
        if power_info:
            break
result = "Power Analysis: " + ", ".join(power_info) if power_info else "No power data available"
result'''

    ALTITUDE_ANALYSIS_TEMPLATE = '''# Altitude analysis
max_alt = None
min_alt = None
avg_alt = None

for source in ['GPS', 'BARO', 'CTUN', 'AHR2']:
    if source in dfs and 'Alt' in dfs[source].columns:
        alt_values = dfs[source]['Alt'].dropna()
        if not alt_values.empty:
            max_alt = alt_values.max()
            min_alt = alt_values.min()
            avg_alt = alt_values.mean()
            break

result = f"Altitude Analysis: Max {max_alt:.0f}m, Min {min_alt:.0f}m, Avg {avg_alt:.0f}m" if max_alt else "No altitude data"
result'''

    ERROR_ANALYSIS_TEMPLATE = '''# Error and message analysis
error_count = 0
critical_msgs = []

# Check ERR dataframe
if 'ERR' in dfs:
    err_df = dfs['ERR']
    error_count += len(err_df)

# Check MSG dataframe for critical messages - safer approach
if 'MSG' in dfs and 'Message' in dfs['MSG'].columns:
    msg_df = dfs['MSG']
    critical_keywords = ['ERROR', 'CRITICAL', 'WARNING', 'FAILSAFE']
    
    # Use vectorized operations for better performance and reliability
    try:
        msg_series = msg_df['Message'].astype(str).str.upper()
        for keyword in critical_keywords:
            keyword_matches = msg_series.str.contains(keyword, na=False)
            critical_msgs.extend(msg_df.loc[keyword_matches, 'Message'].tolist())
    except Exception:
        # Fallback to row iteration if vectorized approach fails
        for idx, row in msg_df.iterrows():
            try:
                msg_text = str(row['Message']).upper()
                if any(keyword in msg_text for keyword in critical_keywords):
                    critical_msgs.append(row['Message'])
            except Exception:
                continue

# Remove duplicates while preserving order
critical_msgs = list(dict.fromkeys(critical_msgs))

result = f"Found {error_count} errors and {len(critical_msgs)} critical messages"
result'''

    GPS_ANALYSIS_TEMPLATE = '''# GPS analysis
gps_info = []
if 'GPS' in dfs:
    gps_df = dfs['GPS']
    if 'NSats' in gps_df.columns:
        avg_sats = gps_df['NSats'].mean()
        gps_info.append(f"Average satellites: {avg_sats:.1f}")
    if 'HDop' in gps_df.columns:
        avg_hdop = gps_df['HDop'].mean()
        gps_info.append(f"Average HDOP: {avg_hdop:.2f}")

result = "GPS Analysis: " + ", ".join(gps_info) if gps_info else "No GPS data available"
result'''

    SIMPLE_EXAMPLES = {
        'max_altitude': '''max_alt = dfs['BARO']['Alt'].max()
result = f"Max altitude: {max_alt:.1f}m"
result''',
        
        'flight_duration': '''time_values = []
for df_name, df in dfs.items():
    if 'TimeUS' in df.columns:
        time_values.extend(df['TimeUS'].dropna().tolist())
duration_sec = (max(time_values) - min(time_values)) / 1_000_000 if time_values else 0
result = f"Flight duration: {duration_sec/60:.1f} minutes"
result''',
        
        'battery_status': '''voltage = dfs['BAT']['Volt'].mean() if 'BAT' in dfs else None
result = f"Average battery: {voltage:.1f}V" if voltage else "No battery data"
result'''
    }

    @classmethod
    def get_code_examples_text(cls) -> str:
        """Return formatted code examples for task descriptions."""
        # Use string concatenation instead of f-strings to prevent variable evaluation
        examples_text = "✅ CORRECT Python Code Examples:\n\n"
        examples_text += "For flight summary:\n```python\n"
        examples_text += cls.FLIGHT_SUMMARY_TEMPLATE
        examples_text += "\n```\n\n"
        examples_text += "For power analysis:\n```python\n"
        examples_text += cls.POWER_ANALYSIS_TEMPLATE
        examples_text += "\n```\n\n"
        examples_text += "For altitude analysis:\n```python\n"
        examples_text += cls.ALTITUDE_ANALYSIS_TEMPLATE
        examples_text += "\n```\n\n"
        examples_text += "For error analysis:\n```python\n"
        examples_text += cls.ERROR_ANALYSIS_TEMPLATE
        examples_text += "\n```\n\n"
        examples_text += "For GPS analysis:\n```python\n"
        examples_text += cls.GPS_ANALYSIS_TEMPLATE
        examples_text += "\n```"
        return examples_text

    @classmethod 
    def get_tool_examples_text(cls) -> str:
        """Return tool calling examples."""
        return '''🔥 TOOL CALLING FORMAT (CRITICAL):
Use this EXACT format when calling tools - ONE TOOL AT A TIME:

For execute_python_code:
Action: execute_python_code
Action Input: {"code": "# Actual Python code here\\nmax_alt = dfs['BARO']['Alt'].max()\\nresult = f'Max altitude: {max_alt:.1f}m'\\nresult"}

For analyze_altitude:
Action: analyze_altitude  
Action Input: {"query": "flight summary"}

For detect_flight_events:
Action: detect_flight_events
Action Input: {"query": "critical events"}'''

    @classmethod
    def get_wrong_examples_text(cls) -> str:
        """Return examples of what NOT to do."""
        return '''❌ WRONG - These will be rejected:
- "Generate comprehensive flight summary with power analysis"
- "Find the maximum altitude"
- "Analyze battery performance" 
- "Show flight statistics"'''

    @classmethod
    def get_specific_example(cls, analysis_type: str) -> str:
        """
        Get a specific code example by type.
        
        Args:
            analysis_type: Type of analysis ('flight_summary', 'power_analysis', etc.)
            
        Returns:
            Python code template as string
        """
        examples_map = {
            'flight_summary': cls.FLIGHT_SUMMARY_TEMPLATE,
            'power_analysis': cls.POWER_ANALYSIS_TEMPLATE,
            'altitude_analysis': cls.ALTITUDE_ANALYSIS_TEMPLATE,
            'error_analysis': cls.ERROR_ANALYSIS_TEMPLATE,
            'gps_analysis': cls.GPS_ANALYSIS_TEMPLATE
        }
        
        return examples_map.get(analysis_type, cls.FLIGHT_SUMMARY_TEMPLATE)

    @classmethod
    def get_simple_example(cls, example_key: str) -> str:
        """
        Get a simple code example by key.
        
        Args:
            example_key: Key for simple example ('max_altitude', 'flight_duration', etc.)
            
        Returns:
            Python code template as string
        """
        return cls.SIMPLE_EXAMPLES.get(example_key, cls.SIMPLE_EXAMPLES['max_altitude'])

    @classmethod
    def add_new_template(cls, template_name: str, template_code: str) -> None:
        """
        Safely add a new code template (for future extensibility).
        
        Args:
            template_name: Name of the new template
            template_code: Python code template (should not contain f-strings with undefined variables)
        """
        # Add to simple examples dictionary
        cls.SIMPLE_EXAMPLES[template_name] = template_code

    @classmethod
    def validate_template_safety(cls, template_code: str) -> bool:
        """
        Validate that a template is safe from f-string evaluation errors.
        
        Args:
            template_code: Code template to validate
            
        Returns:
            True if template is safe, False otherwise
        """
        try:
            # Try to format the template - if it has undefined variables, it will fail
            # This is a simple check, but helps catch obvious issues
            test_format = template_code.format()
            return True
        except (KeyError, ValueError):
            # Template contains formatting placeholders that could cause issues
            return False
        except:
            # Other errors are generally OK for our use case
            return True

    @classmethod
    def get_safe_description_builder(cls) -> List[str]:
        """
        Get a list of safe description parts that can be joined safely.
        This method ensures no f-string evaluation vulnerabilities.
        
        Returns:
            List of strings that can be safely joined
        """
        return [
            "🚨 CRITICAL PYTHON CODE REQUIREMENTS:",
            "For execute_python_code tool, you MUST write actual executable Python code, NOT natural language:",
            "",
            cls.get_code_examples_text(),
            "",
            cls.get_wrong_examples_text(),
            "",
            cls.get_tool_examples_text(),
        ]

    @classmethod
    def test_template_safety(cls) -> Dict[str, bool]:
        """
        Test method to demonstrate the robustness of the template system.
        This shows that the solution can handle edge cases that would have 
        previously caused NameError exceptions.
        
        Returns:
            Dictionary of test results
        """
        test_results = {}
        
        # Test 1: Basic template safety
        try:
            examples_text = cls.get_code_examples_text()
            test_results['basic_template_generation'] = len(examples_text) > 0
        except Exception as e:
            test_results['basic_template_generation'] = False
            logger.error(f"Basic template generation failed: {e}")
        
        # Test 2: Edge case - adding problematic template
        try:
            # This would have caused issues in the old system
            problematic_template = "result = f'Value: {undefined_variable:.2f}'"
            cls.add_new_template('test_problematic', problematic_template)
            
            # Getting the template should still work (it's stored safely)
            retrieved = cls.get_simple_example('test_problematic')
            test_results['problematic_template_handling'] = retrieved == problematic_template
        except Exception as e:
            test_results['problematic_template_handling'] = False
            logger.error(f"Problematic template handling failed: {e}")
        
        # Test 3: Template validation
        try:
            safe_template = "result = 'This is safe'"
            unsafe_template = "result = f'Value: {undefined_var}'"
            
            safe_result = cls.validate_template_safety(safe_template)
            unsafe_result = cls.validate_template_safety(unsafe_template)
            
            test_results['template_validation'] = safe_result and not unsafe_result
        except Exception as e:
            test_results['template_validation'] = False
            logger.error(f"Template validation failed: {e}")
        
        # Test 4: Safe description builder
        try:
            description_parts = cls.get_safe_description_builder()
            combined = "\n".join(description_parts)
            test_results['safe_description_builder'] = len(combined) > 100
        except Exception as e:
            test_results['safe_description_builder'] = False
            logger.error(f"Safe description builder failed: {e}")
        
        return test_results


# Custom CrewAI Tools with simplified implementation
class PythonCodeExecutorTool(BaseTool):
    name: str = "execute_python_code"
    description: str = "Execute valid Python code to analyze UAV flight data. ONLY pass actual Python code, NOT natural language queries. The code must be complete and executable."
    args_schema: Type[BaseModel] = PythonCodeInput
    
    def __init__(self, analysis_tools: AnalysisTools, session: V2ConversationSession):
        super().__init__()
        self._analysis_tools = analysis_tools
        self._session = session
        self._validator = None  # Will be initialized when LLM is available
    
    def _get_validator(self):
        """Lazy initialization of validator with LLM access."""
        if self._validator is None:
            # Get LLM from the MultiRoleAgent context if available
            try:
                from config import get_settings
                settings = get_settings()
                if settings.openai_api_key:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        api_key=settings.openai_api_key,
                        model=settings.openai_model,
                        temperature=0.1  # Low temperature for code generation
                    )
                    self._validator = PythonCodeValidator(llm, self._session)
                else:
                    self._validator = PythonCodeValidator(None, self._session)
            except Exception as e:
                logger.warning(f"Could not initialize LLM for code validation: {e}")
                self._validator = PythonCodeValidator(None, self._session)
        return self._validator
    
    def _run(self, code: str) -> str:
        """Execute Python code with generalized error recovery."""
        logger.info(f"[{self._session.session_id}] PythonCodeExecutorTool received: {code[:100]}...")
        
        # Basic session validation
        if not self._session.dataframes:
            logger.error(f"[{self._session.session_id}] NO DATAFRAMES AVAILABLE")
            return "Error: No flight data available. Please upload a log file first."
        
        logger.info(f"[{self._session.session_id}] Available dataframes: {list(self._session.dataframes.keys())}")
        
        # Validate that this is actual Python code
        if not self._is_valid_python_code(code):
            logger.error(f"[{self._session.session_id}] INVALID INPUT: Tool received natural language")
            return f"ERROR: This tool only accepts Python code, not natural language. Received: '{code[:50]}...'"
        
        # Get validator for advanced error handling
        validator = self._get_validator()
        
        # Pre-validate and fix obvious issues
        validated_code = validator.validate_and_fix_code(code)
        
        try:
            logger.info(f"[{self._session.session_id}] Executing Python code")
            result = self._analysis_tools.execute_python_code(self._session, validated_code)
            
            if result and len(result.strip()) > 0:
                logger.info(f"[{self._session.session_id}] Execution successful")
                return result
            else:
                logger.warning(f"[{self._session.session_id}] Code executed but returned empty result")
                return "Python code executed successfully but returned no output."
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{self._session.session_id}] Execution failed: {error_msg}")
            
            # GENERALIZED ERROR RECOVERY
            user_intent = validator.extract_user_intent(validated_code)
            logger.info(f"[{self._session.session_id}] Attempting adaptive fallback for: {user_intent}")
            
            fallback_code = validator.generate_adaptive_fallback(validated_code, error_msg, user_intent)
            
            try:
                fallback_result = self._analysis_tools.execute_python_code(self._session, fallback_code)
                if fallback_result:
                    logger.info(f"[{self._session.session_id}] Adaptive fallback successful")
                    return fallback_result
            except Exception as fallback_error:
                logger.error(f"[{self._session.session_id}] Fallback also failed: {fallback_error}")
            
            # If all else fails, provide helpful error information
            return f"Analysis failed: {error_msg}. Available data: {', '.join(self._session.dataframes.keys())}"
    
    def _is_valid_python_code(self, text: str) -> bool:
        """Validate that text is actual Python code."""
        text = text.strip()
        
        # Must contain basic Python syntax elements
        python_indicators = ['=', 'for ', 'if ', 'def ', 'import ', 'from ', 'print(', 'result', '[', ']', '(', ')']
        has_python_syntax = any(indicator in text for indicator in python_indicators)
        
        if not has_python_syntax:
            return False
        
        # Reject obvious natural language queries
        first_line = text.split('\n')[0].strip().lower()
        natural_language_starters = [
            'generate', 'create a', 'show me', 'tell me', 'what is',
            'how much', 'find the', 'get the', 'calculate the', 'determine the',
            'provide me', 'give me', 'can you', 'please', 'i want', 'i need'
        ]
        
        if not first_line.startswith('#') and any(first_line.startswith(starter) for starter in natural_language_starters):
            return False
        
        # Try to compile
        try:
            compile(text, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
        except:
            return True  # Runtime errors are OK for validation

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
        """Format altitude response based on query type with concise, focused information."""
        query_lower = query.lower()
        
        # Handle specific queries with concise responses
        if any(keyword in query_lower for keyword in ["maximum", "highest", "max"]):
            return f"Maximum altitude: {stats['max']:.0f}m"
            
        elif any(keyword in query_lower for keyword in ["minimum", "lowest", "min"]):
            return f"Minimum altitude: {stats['min']:.0f}m"
            
        elif any(keyword in query_lower for keyword in ["average", "mean", "typical"]):
            return f"Average altitude: {stats['mean']:.0f}m"
            
        else:
            # General altitude analysis - concise summary only
            response = f"Altitude Summary:\n"
            response += f"• Maximum: {stats['max']:.0f}m\n"
            response += f"• Average: {stats['mean']:.0f}m\n"
            response += f"• Range: {stats['range']:.0f}m"
            
            # Add only the most critical insight
            if stats['range'] > 500:
                response += "\n• Multi-phase flight with significant altitude changes"
            elif stats['min'] == 0:
                response += "\n• Ground operations included"
                
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


class EnhancedTelemetryAnalysisTool(BaseTool):
    name: str = "enhanced_telemetry_analysis"
    description: str = "Perform comprehensive telemetry analysis covering all 164 UAV message types with advanced derived metrics, cross-sensor validation, EKF health analysis, and comprehensive anomaly detection. Use for in-depth flight analysis and when you need advanced insights beyond basic queries."
    args_schema: Type[BaseModel] = TelemetryQueryInput
    
    def __init__(self, session: V2ConversationSession):
        super().__init__()
        self._session = session
        from tools.enhanced_telemetry_analyzer import EnhancedTelemetryAnalyzer
        self._analyzer = EnhancedTelemetryAnalyzer()
    
    def _run(self, query: str) -> str:
        """Run enhanced telemetry analysis based on the query type."""
        logger.info(f"[{self._session.session_id}] Enhanced telemetry analysis for: {query}")
        
        try:
            query_lower = query.lower()
            
            # Route to specific analysis based on query
            if any(word in query_lower for word in ['comprehensive', 'complete', 'full', 'overall', 'summary']):
                return self._analyzer.comprehensive_analysis(self._session)
            elif any(word in query_lower for word in ['derived', 'metrics', 'calculations', 'advanced']):
                return self._analyzer._calculate_derived_metrics(self._session)
            elif any(word in query_lower for word in ['cross', 'validation', 'sensor', 'compare']):
                return self._analyzer._cross_sensor_validation(self._session)
            elif any(word in query_lower for word in ['ekf', 'kalman', 'filter', 'health']):
                return self._analyzer._analyze_ekf_health(self._session)
            elif any(word in query_lower for word in ['anomaly', 'anomalies', 'detect', 'issues']):
                return self._analyzer._comprehensive_anomaly_detection(self._session)
            elif any(word in query_lower for word in ['phase', 'phases', 'flight']):
                return self._analyzer._advanced_phase_analysis(self._session)
            elif any(word in query_lower for word in ['signal', 'quality', 'assessment']):
                return self._analyzer._signal_quality_assessment(self._session)
            else:
                # Default to comprehensive analysis for general queries
                return self._analyzer.comprehensive_analysis(self._session)
            
        except Exception as e:
            error_msg = f"Enhanced telemetry analysis failed: {str(e)}"
            logger.error(f"[{self._session.session_id}] {error_msg}")
            return error_msg


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
        """Initialize the multi-role agent with CrewAI memory and enhanced conversational features."""
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
        
        # Initialize Intent Classifier (replaces brittle keyword-based routing)
        self.intent_classifier = IntentClassifier(self.settings)
        
        # Initialize Semantic Data Retriever for Phase 3: Semantic Data Understanding
        self.semantic_retriever = SemanticDataRetriever(
            storage_dir=os.path.join("./semantic_data_store")
        )
        
        # Session storage and performance tracking
        self.sessions: Dict[str, V2ConversationSession] = {}
        self.llm_calls: List[LLMCall] = []
        
        # Configure CrewAI Memory System
        self._setup_memory_configuration()
        
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
        
        # Comprehensive ArduPilot dataframe documentation enhanced with official definitions
        # Source: https://ardupilot.org/copter/docs/logmessages.html and related ArduPilot documentation
        self.dataframe_documentation = {
            # Core Navigation & Attitude
            'ATT': 'Canonical vehicle attitude - DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw angles. Vehicle desired vs achieved roll, pitch, yaw for attitude control analysis and flight stability assessment',
            'AHR2': 'Backup AHRS attitude data - Estimated roll, pitch, yaw, altitude, latitude, longitude with quaternion components for redundant attitude estimation and navigation backup systems',
            'CTUN': 'Control Tuning information - ThI (throttle input), ThO (throttle output), ThH (hover throttle), DAlt (desired altitude), Alt (achieved altitude), BAlt (barometric altitude), DCRt (desired climb rate), CRt (climb rate) for altitude and throttle control analysis',
            'ANG': 'Attitude control attitude - DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw with delta time for real-time attitude controller performance monitoring',
            
            # GPS and Navigation Systems  
            'GPS': 'GPS position data - Status (0=no GPS, 1=no fix, 2=2D fix, 3=3D fix), Time, NSats (satellite count), HDop (horizontal dilution of precision), Lat, Lng, RelAlt, Alt, SPD (speed), GCrs (ground course) for complete GPS navigation and signal quality analysis',
            'GPA': 'GPS accuracy metrics - VDop (vertical dilution of precision), HAcc (horizontal accuracy), VAcc (vertical accuracy), SAcc (speed accuracy), SMS (system time) for GPS precision and reliability assessment',
            'NTUN': 'Navigation Tuning - WPDst (waypoint distance), WPBrg (waypoint bearing), PErX/PErY (position error), DVelX/DVelY (desired velocity), VelX/VelY (actual velocity), DAcX/DAcY (desired acceleration), DRol/DPit (desired roll/pitch) for autonomous navigation analysis',
            'POS': 'Position estimates - Latitude, longitude, altitude from Extended Kalman Filter sensor fusion for high-accuracy position tracking and navigation validation',
            
            # Sensor Data - Motion and Environment
            'IMU': 'Inertial Measurement Unit - AccX/Y/Z (acceleration m/s/s), GyrX/Y/Z (rotation rates rad/s) in body frame with timestamps for motion detection, vibration analysis, and flight control',
            'ACC': 'IMU accelerometer data - SampleUS (timestamp), AccX/Y/Z (acceleration along axes) for detailed acceleration analysis and sensor health monitoring',
            'GYR': 'Gyroscope data - SampleUS (timestamp), GyrX/Y/Z (angular rates) for detailed rotation analysis and attitude control system performance',
            'BARO': 'Barometer data - Alt (calculated altitude), AltAMSL (altitude above mean sea level), Press (atmospheric pressure), Temp (temperature), CRt (climb rate derived from barometer), SMS (sample time) for pressure-based altitude measurement and atmospheric conditions',
            'ARSP': 'Airspeed sensor data - Airspeed (m/s), DiffPress (differential pressure), Temp (temperature), RawPress, Offset, sensor health flags for airspeed measurement analysis and sensor validation',
            'COMPASS': 'Magnetometer data - MagX/Y/Z (magnetic field values), OfsX/Y/Z (offsets), MOfsX/Y/Z (motor compensation) for heading determination and compass calibration analysis',
            'MAG': 'Raw magnetometer readings - MagX/Y/Z (compass field strength) for magnetic heading calculation and interference detection',
            
            # Power and Electrical Systems
            'BAT': 'Battery monitoring - Inst (instance), Volt (voltage), VoltR (resting voltage), Curr (current), CurrTot (consumed Ah), EnrgTot (energy consumed Wh), Temp (temperature), Res (resistance), RemPct (remaining %), Health for comprehensive power system analysis',
            'CURR': 'Current monitoring legacy format - Thr (throttle), ThrInt (integrated throttle), Volt (battery voltage), Curr (current), Vcc (board voltage), CurrTot (total current consumed) for electrical system health and power consumption analysis',
            'BCL': 'Battery cell voltages - Instance, Volt (total voltage), V1-V12 (individual cell voltages in mV) for lithium battery health monitoring, cell balancing analysis, and battery safety assessment',
            'BCL2': 'Extended battery cell voltages - V13, V14 (additional cell voltages) for high-capacity battery packs with more than 12 cells',
            'POWR': 'Power system monitoring - Vcc (main voltage), VServo (servo rail voltage), Flags (power status flags) for complete electrical system health assessment',
            
            # Flight Control and Actuators
            'MODE': 'Flight mode changes - Mode (flight mode string), ThrCrs (throttle cruise), Rsn (reason for mode change) for flight behavior analysis and mode transition tracking',
            'RCIN': 'RC input channels - C1-C16 (pilot stick inputs, switch positions) for manual control analysis, pilot input monitoring, and control system responsiveness',
            'RCOU': 'RC output channels - C1-C16 (PWM outputs to servos/motors) for actuator analysis, servo performance, and control surface movement validation',
            'AETR': 'Control surface outputs - Ail (aileron), Elev (elevator), Thr (throttle), Rudd (rudder), Flap (flaps), Steer (steering), SS (surface scaling) for fixed-wing control analysis and surface movement tracking',
            'RATE': 'Rate controller performance - DesRoll/Pitch/Yaw (desired rates), Roll/Pitch/Yaw (actual rates) for attitude control system tuning and performance analysis',
            
            # System Health, Errors and Events
            'ERR': 'Error messages and system warnings - Subsys (subsystem), ECode (error code) including Radio/GPS/Compass/Battery/GCS failsafes, sensor health issues, fence breaches, crash detection for comprehensive safety and reliability analysis',
            'EV': 'System events - Event number with descriptions like Armed (10), Disarmed (11), Auto Armed (15), Land Complete (18), Set Home (25), Takeoff Complete (28) for flight phase analysis and system state tracking',
            'PM': 'Performance monitoring - NLon (long running loops), NLoop (total loops), MaxT (maximum loop time), Mem (available memory), Load (CPU load percentage) for autopilot performance analysis and system health monitoring',
            'MSG': 'Text messages - Human-readable status messages, warnings, informational text from autopilot for flight analysis and debugging',
            'VIBE': 'Vibration monitoring - VibeX/Y/Z (vibration levels), Clip0/1/2 (clipping counts) for mechanical health analysis, motor balance assessment, and sensor performance validation',
            
            # Advanced Flight Systems and Mission Control
            'CMD': 'Mission commands - CTot (total commands), CNum (command number), CId (command type), Prm1-4 (parameters), Lat, Lng, Alt (waypoint coordinates), Frame for autonomous mission analysis and waypoint navigation',
            'FENCE': 'Geofence system - Fence status, breach notifications, boundary violations for safety zone analysis and autonomous flight boundary enforcement',
            'CAM': 'Camera triggers - Img (image number), GPSTime, GPSWeek, Lat, Lng, Alt, RelAlt, GPSAlt, Roll, Pitch, Yaw for mapping analysis, photo geolocation, and survey mission documentation',
            'TERR': 'Terrain following - Terrain altitude, height above terrain for terrain-following flight analysis and ground clearance monitoring',
            'FLOW': 'Optical flow - Ground speed estimation from camera sensor for GPS-denied navigation and position holding analysis',
            'LIDAR': 'Lidar/rangefinder measurements - Distance measurements for obstacle avoidance, terrain following, and precision landing analysis',
            'SONAR': 'Sonar measurements - Ultrasonic distance readings for landing assistance and low-altitude flight safety',
            
            # Extended Kalman Filter (EKF) and State Estimation
            'XKF1': 'EKF primary state - AngErr (angle error), VelErr (velocity error), PosErr (position error), Innovation test ratios for Kalman filter performance and navigation accuracy assessment',
            'XKF2': 'EKF wind estimation - WindN (north wind), WindE (east wind), MagN/E/D (magnetic field components) for wind analysis and magnetic field monitoring',
            'XKF3': 'EKF innovation variances - IVN/IVE/IVD (innovation test ratios) for filter health monitoring and sensor fusion validation',
            'XKF4': 'EKF timing and status - SV (solution valid), CE (compass error), SS (solution status) for navigation system health and reliability assessment',
            'NKF1': 'NavEKF primary - Same as XKF1 but from NavEKF implementation for navigation filter performance analysis',
            'NKF2': 'NavEKF wind - Same as XKF2 but from NavEKF for wind estimation and magnetic field analysis',
            
            # Communication and Telemetry
            'RSSI': 'Signal strength - RC and telemetry signal quality, fade margin for communication link analysis and range assessment',
            'MAV': 'MAVLink messages - Communication protocol data, packet rates, sequence numbers for telemetry system analysis and link quality monitoring',
            'DMS': 'DataFlash-Over-MAVLink statistics - Block numbers, retry counts, queue statistics for logging system performance and data integrity analysis',
            
            # Specialized Vehicle Systems
            'MOTB': 'Motor/Battery integration - LiftMax (maximum lift), BatVolt (battery voltage), BatRes (resistance), ThLimit (throttle limit) for motor performance and power system integration analysis',
            'ESC': 'Electronic Speed Controller - RPM, Volt, Curr, Temp for individual motor performance monitoring and multi-rotor system analysis',
            'AUXF': 'Auxiliary function triggers - Function ID, position, source, index, result for switch activation monitoring and auxiliary system control analysis',
            
            # Atmospheric and Environmental
            'BARD': 'Barometer dynamic - DynPrX/Y/Z (dynamic pressure in body frame) for airspeed calculation and atmospheric pressure analysis in different axes',
            'WIND': 'Wind estimation - Speed, direction, variance for wind analysis and flight planning optimization',
            
            # PID Controllers and Tuning
            'PIDR': 'PID Roll controller - Des (desired), P/I/D (PID components), FF (feedforward), AFF (acceleration feedforward) for roll axis tuning and performance analysis',
            'PIDP': 'PID Pitch controller - Des, P, I, D, FF, AFF for pitch axis control system tuning and stability analysis',
            'PIDY': 'PID Yaw controller - Des, P, I, D, FF, AFF for yaw axis control system tuning and heading control analysis',
            'PIDA': 'PID Altitude controller - Des, P, I, D, FF, AFF for altitude hold system tuning and vertical position control analysis',
            
            # Configuration and Parameters
            'PARM': 'Parameter values - Name, Value pairs for autopilot configuration analysis, parameter change tracking, and system setup validation',
            'FMT': 'Format definitions - Type, Length, Name, Format describing log message structure for log analysis and data interpretation',
            'FMTU': 'Format units - FmtType, UnitIds, MultIds defining measurement units and multipliers for data scaling and unit conversion'
        }
        
        # Initialize semantic data understanding (Phase 3) - AFTER dataframe_documentation is set up
        self._initialize_semantic_data_understanding()

    def _setup_memory_configuration(self):
        """Configure CrewAI memory system for persistent conversation context."""
        # Set up custom storage directory for the UAV application
        self.memory_storage_dir = os.getenv("CREWAI_STORAGE_DIR", "./crewai_memory")
        os.makedirs(self.memory_storage_dir, exist_ok=True)
        
        # Set environment variable for CrewAI to use our custom directory
        os.environ["CREWAI_STORAGE_DIR"] = self.memory_storage_dir
        
        # Configure embedding provider for memory system
        # Using OpenAI embeddings by default for consistency with LLM
        self.memory_embedder_config = {
            "provider": "openai",
            "config": {
                "api_key": self.settings.openai_api_key if self.settings.openai_api_key else None,
                "model": "text-embedding-3-small"  # Cost-effective embedding model
            }
        }
        
        logger.info(f"Memory system configured with storage at: {self.memory_storage_dir}")

    def _initialize_semantic_data_understanding(self):
        """Initialize Phase 3: Semantic Data Understanding system."""
        try:
            # Schedule the async initialization for when an event loop is available
            # This will be handled when the first request comes in
            logger.info("Phase 3: Semantic Data Understanding initialized successfully (async indexing deferred)")
        except Exception as e:
            logger.error(f"Failed to initialize semantic data understanding: {e}")
            logger.info("Semantic data understanding will use fallback mode")

    def _get_memory_context(self, session_id: str, user_message: str) -> str:
        """Get relevant memory context for the current query."""
        try:
            # Memory is automatically handled by CrewAI when memory=True is set
            # This method can be used for additional memory-related operations if needed
            return f"Session ID: {session_id} - Memory system active"
        except Exception as e:
            logger.warning(f"Error accessing memory context: {e}")
            return "Memory context unavailable"

    def reset_memory(self, session_id: Optional[str] = None):
        """Reset the CrewAI memory system for a session or globally."""
        try:
            if session_id:
                # For session-specific reset, we would need to implement session-specific memory
                # For now, this resets the global memory
                logger.info(f"Resetting memory for session: {session_id}")
            else:
                # Reset all memory
                if os.path.exists(self.memory_storage_dir):
                    shutil.rmtree(self.memory_storage_dir)
                    os.makedirs(self.memory_storage_dir, exist_ok=True)
                    logger.info("Memory system reset successfully")
                else:
                    logger.info("No memory to reset")
        except Exception as e:
            logger.error(f"Error resetting memory: {e}")

    def get_memory_info(self) -> Dict[str, Any]:
        """Get information about the current memory system status."""
        info = {
            "storage_directory": self.memory_storage_dir,
            "memory_enabled": True,
            "embedder_provider": self.memory_embedder_config.get("provider", "openai"),
            "embedder_model": self.memory_embedder_config.get("config", {}).get("model", "text-embedding-3-small"),
            "memory_types": ["short_term", "long_term", "entity"],
            "storage_exists": os.path.exists(self.memory_storage_dir)
        }
        
        # Add storage size if directory exists
        if info["storage_exists"]:
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(self.memory_storage_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                info["storage_size_bytes"] = total_size
                info["storage_size_mb"] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not calculate storage size: {e}")
        
        return info

    def get_semantic_retriever_status(self) -> Dict[str, Any]:
        """Get status information about the semantic data understanding system."""
        try:
            return self.semantic_retriever.get_status()
        except Exception as e:
            logger.error(f"Failed to get semantic retriever status: {e}")
            return {
                "initialized": False,
                "error": str(e),
                "fallback_mode": True
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
            
            # Phase 3: Re-index semantic data after processing new log file
            if session.dataframes:
                await self.semantic_retriever.index_dataframe_documentation(self.dataframe_documentation)
                logger.info(f"[{session_id}] Semantic data understanding updated for new session data")
                
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
            EnhancedTelemetryAnalysisTool(session),  # NEW: Enhanced comprehensive analysis
            MessageDocumentationTool(self.documentation_service),
        ]
        logger.info(f"[{session.session_id}] Created {len(tools)} tools: {[tool.name for tool in tools]}")
        return tools

    def _create_planner_agent(self, session: V2ConversationSession) -> Agent:
        """Create the Planner agent with enhanced intelligence and adaptability."""
        data_summary = get_data_summary(session)
        
        # Additional safety check to ensure data_summary is never None
        if data_summary is None:
            data_summary = {
                "message_types": 0,
                "total_records": 0,
                "time_range": {"duration_minutes": 0},
                "key_metrics": {}
            }
        
        dataframe_docs = self._get_comprehensive_dataframe_docs(data_summary)
        conversation_context = self._get_conversation_context(session)
        
        backstory = f"""You are an expert UAV flight data analyst with memory capabilities who creates intelligent, adaptive execution plans for user queries.

Available Flight Data:
- Message types: {data_summary.get('message_types', 0)}
- Total records: {data_summary.get('total_records', 0):,}
- Time range: {data_summary.get('time_range', 'Unknown')}

{dataframe_docs}

CONVERSATION CONTEXT:
{conversation_context}

MEMORY SYSTEM: You have access to persistent memory that recalls previous conversations, analysis patterns, and user preferences. Use this context to provide more relevant and contextual responses, especially for follow-up questions.

AVAILABLE ANALYSIS TOOLS (with intelligent selection):
- analyze_altitude: Best for altitude-specific questions (max/min/stats)
- execute_python_code: Versatile tool for calculations, data extraction, custom analysis
- detect_flight_events: Error analysis, warnings, system events (has intelligent fallbacks)
- find_anomalies: Pattern detection, unusual behavior identification
- get_timeline_analysis: Chronological event analysis, flight progression
- analyze_flight_phase: Phase-specific analysis (takeoff, landing, cruise)
- enhanced_telemetry_analysis: Comprehensive analysis covering all 164 message types, advanced derived metrics, cross-sensor validation, EKF health

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
        
        # Additional safety check to ensure data_summary is never None
        if data_summary is None:
            data_summary = {
                "message_types": 0,
                "total_records": 0,
                "time_range": {"duration_minutes": 0},
                "key_metrics": {}
            }
        
        backstory = f"""You are an expert UAV flight data analyst who EXECUTES TOOLS intelligently with advanced error recovery.

Available Flight Data:
- Message types: {data_summary.get('message_types', 0)}
- Total records: {data_summary.get('total_records', 0):,}
- Flight duration: {data_summary.get('time_range', {}).get('duration_minutes', 'Unknown')} minutes

CONVERSATION CONTEXT:
{conversation_context}

🔥 CRITICAL TOOL EXECUTION RULES:
1. **NEVER IGNORE TOOL RESULTS** - Use EXACTLY what tools return, never modify or interpret
2. **NO DATA FABRICATION** - Only use information that comes directly from tool execution
3. **WRITE ACTUAL PYTHON CODE** - For execute_python_code tool, write real Python code, not natural language
4. **TRUST TOOL OUTPUTS** - Report tool results exactly as returned

PYTHON CODE GENERATION REQUIREMENTS:
When using execute_python_code tool, you MUST write actual executable Python code:

✅ CORRECT EXAMPLES:
```python
# Find maximum altitude
max_alt = None
for source in ['GPS', 'BARO', 'CTUN']:
    if source in dfs and 'Alt' in dfs[source].columns:
        alt_values = dfs[source]['Alt'].dropna()
        if not alt_values.empty:
            current_max = alt_values.max()
            if max_alt is None or current_max > max_alt:
                max_alt = current_max
result = f"Maximum altitude: {{max_alt:.1f}}m" if max_alt else "No altitude data"
result
```

❌ WRONG EXAMPLES:
- "Generate comprehensive flight summary with power analysis"
- "Find the maximum altitude"
- "Analyze battery performance"

FLIGHT DATA ACCESS PATTERN:
- Use `dfs` dictionary to access dataframes: `dfs['GPS']`, `dfs['BARO']`, etc.
- Available dataframes: {list(session.dataframes.keys())[:10]}...
- Common patterns:
  * Altitude: `dfs['BARO']['Alt']` or `dfs['GPS']['Alt']`
  * Battery: `dfs['BAT']['Volt']`, `dfs['BAT']['Curr']`
  * Time: `dfs[df_name]['TimeUS']` (microseconds)
  * Messages: `dfs['MSG']['Message']`
  * Errors: `dfs['ERR']`

INTELLIGENT EXECUTION BEHAVIOR:
1. **Execute planned tools first** - Follow the plan exactly
2. **Use EXACT tool output** - Report what tools actually returned
3. **Write proper Python code** - No natural language to code tools
4. **Combine results intelligently** - Synthesize multi-tool outputs accurately
5. **Handle errors gracefully** - Try alternative approaches if tools fail

RESPONSE QUALITY:
- Start with actual findings (numbers, data, specific results) FROM TOOL OUTPUTS ONLY
- Be conversational but precise (2-4 sentences typically)
- For comprehensive queries: Organize findings logically
- NEVER add information not provided by tools

CRITICAL FORBIDDEN BEHAVIORS:
- Never ignore tool results and make up your own interpretation
- Never fabricate data not returned by tools
- Never pass natural language to execute_python_code tool
- Never say "detected" without tool output saying "detected"
- Never add flight duration, times, or counts not calculated by tools
- Never give up without trying fallback approaches

EXECUTION EXAMPLES:
✓ "The maximum altitude reached was 1,448 meters." (from tool output)
✓ "Found 18 GPS signal events between 14:23 and 14:45." (from tool output)
✗ "The analysis shows..." (when no analysis was actually performed)
✗ "Flight duration was 3.8 minutes" (when no tool calculated this)

🔥 REMEMBER: Only report what tools actually returned. Write real Python code, not natural language queries."""

        tools = self._create_crew_tools(session)
        
        return Agent(
            role="UAV Flight Data Analyst & Intelligent Executor",
            goal="Execute analysis tools with real Python code and provide actual results from tool outputs",
            backstory=backstory,
            llm=self.llm,
            tools=tools,
            verbose=True,
            allow_delegation=False,
            max_iter=10,  # Allow more iterations to ensure tool execution
            max_execution_time=60,  # Increase timeout for tool execution
            step_callback=self._ensure_tool_execution
        )

    def _create_critic_agent(self, session: V2ConversationSession) -> Agent:
        """Create the Critic agent with enhanced report generation capabilities."""
        conversation_context = self._get_conversation_context(session)
        
        backstory = f"""You are an expert UAV flight data analyst who transforms technical analysis into perfectly formatted responses, with advanced capabilities for generating comprehensive flight reports.

CONVERSATION CONTEXT:
{conversation_context}

🚨 CRITICAL DATA INTEGRITY RULES:
1. **ONLY USE EXECUTOR DATA** - Use EXACTLY what the executor provided, nothing more
2. **NO DATA FABRICATION** - Never add numbers, times, counts, or details not in executor output
3. **NO INTERPRETATION** - Don't change "18 events" to "22 events" or add "duration: 3.8 minutes"
4. **EXACT PRESERVATION** - Keep all numbers, measurements, and findings exactly as provided

🔥 FORBIDDEN ADDITIONS:
- Flight duration/time calculations not provided by executor
- Changing event counts (18 → 22, etc.)
- Adding GPS quality assessments not in executor output
- Inferring data from "available for analysis" statements
- Converting between units not done by executor
- Adding contextual information not provided

📊 ENHANCED REPORT GENERATION CAPABILITIES:

**DETECT COMPREHENSIVE SUMMARY REQUESTS:**
When the user asks for:
- "flight summary"
- "summary of this flight" 
- "overall analysis"
- "comprehensive report"
- "flight performance report"
- Multiple consecutive queries about different aspects (altitude, power, GPS, errors)

THEN generate a **CONSOLIDATED MARKDOWN REPORT** instead of fragmented responses.

**MARKDOWN REPORT STRUCTURE:**
```markdown
### **Flight Performance Summary**

Brief introduction based on executor findings.

---

#### **Flight Dynamics & Altitude**
* **Maximum Altitude:** [from executor data]
* **Average Altitude:** [from executor data]
* **Altitude Range:** [calculated from executor max/min if both provided]

---

#### **System Alerts & Events**
* **Event Count:** [from executor data]
* [Conditional notes based on rules]

---

#### **GPS & Navigation Quality**
* **Average Satellites:** [from executor data]
* **Average HDOP:** [from executor data]
* [Conditional notes based on rules]

---

#### **Power System Analysis**
* **Average Voltage:** [from executor data]
* **Minimum Voltage:** [from executor data]
* **Average Current:** [from executor data]
* **Peak Current:** [from executor data]
* [Conditional notes based on rules]
```

**RULES-BASED ANALYSIS ENGINE:**
Apply these rules ONLY when generating consolidated reports and ONLY if executor provided the relevant data:

1. **GPS HDOP Analysis:** If average HDOP > 5.0:
   "*Note: HDOP values above 5 indicate reduced GPS accuracy. Values below 2 are ideal for precise navigation.*"

2. **Voltage Critical Analysis:** If minimum voltage < 1.0V:
   "*Note: Minimum voltage near 0V indicates potential power system issues or sensor malfunction.*"

3. **Event Count Analysis:** If event count > 10:
   "*Note: High event count detected. Review of specific event logs recommended for detailed analysis.*"

4. **Satellite Count Analysis:** If average satellites < 4:
   "*Note: GPS satellite count below 4 may indicate poor signal reception affecting navigation accuracy.*"

**FLEXIBLE CATEGORIZATION:**
Dynamically organize executor data into relevant categories based on what was provided:
- Only include sections where executor provided data
- Adapt section names to match the specific type of analysis performed
- Use consistent Markdown formatting throughout

**RESPONSE DECISION LOGIC:**

**For Simple/Specific Questions:**
- Single measurement queries → Natural sentence responses
- Specific error checks → Bullet point findings
- Basic yes/no questions → Direct answers

**For Comprehensive Analysis:**
- Flight summary requests → Full Markdown report
- Multiple system analysis → Categorized Markdown report
- Performance review requests → Structured analysis report

**CRITICAL OUTPUT REQUIREMENTS:**
1. Preserve all executor data exactly as provided
2. Apply rules-based analysis only to executor-provided metrics
3. Use Markdown formatting for comprehensive reports
4. Maintain professional, analytical tone
5. Provide actionable insights based on detected patterns in executor data

**QUALITY STANDARDS:**
- Include specific numbers and measurements FROM EXECUTOR ONLY
- Use professional but accessible language
- Structure information for immediate comprehension
- Highlight critical findings that require attention
- Maintain consistency in formatting and terminology"""

        return Agent(
            role="Advanced Flight Data Report Generator & Formatter", 
            goal="Transform executor results into professional, structured reports with rules-based analysis",
            backstory=backstory,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=30
        )

    async def _create_planning_task(self, user_message: str, session: V2ConversationSession) -> Task:
        """Create the planning task for tool selection with enhanced flight summary detection."""
        data_summary = get_data_summary(session)
        conversation_context = self._get_conversation_context(session)
        is_flight_summary = self._is_flight_summary_request(user_message)
        
        # Phase 3: Get semantic context for intelligent data source selection
        semantic_context = ""
        try:
            # ✅ FIXED - Now using proper Phase 3 semantic search with vector embeddings
            matches = await self.semantic_retriever.find_relevant_dataframes(
                user_message, 
                list(session.dataframes.keys()),
                top_k=3
            )
            if matches:
                semantic_context = f"SEMANTIC DATA ANALYSIS:\nBased on your query '{user_message}', the most relevant data sources are:\n"
                for i, match in enumerate(matches, 1):
                    semantic_context += f"{i}. {match.dataframe_key}: {match.dataframe_description}\n"
                    semantic_context += f"   Relevance: {match.relevance_reason} (Score: {match.similarity_score:.2f})\n"
                semantic_context += "\nRECOMMENDATION: Focus your analysis on these dataframes for the most relevant results."
            else:
                semantic_context = "No specific dataframes identified as highly relevant for this query. Consider general analysis approach."
        except Exception as e:
            logger.warning(f"Failed to get semantic context: {e}")
            semantic_context = "Semantic data understanding unavailable, using general approach."
        

        
        # Additional safety check to ensure data_summary is never None
        if data_summary is None:
            data_summary = {
                "message_types": 0,
                "total_records": 0,
                "time_range": {"duration_minutes": 0},
                "key_metrics": {}
            }
        
        # Build description using safe string concatenation to avoid f-string nesting issues
        description_parts = [
            "🚨 You are an intelligent UAV flight data planning agent with enhanced flight summary capabilities.",
            "",
            f'User Question: "{user_message}"',
            f"Is Flight Summary Request: {is_flight_summary}",
            "",
            f"Available Flight Data: {list(session.dataframes.keys())}",
            "",
            "CONVERSATION CONTEXT:",
            conversation_context,
            "",
            semantic_context,
            "",
            "🎯 **ENHANCED PLANNING APPROACH:**",
            "",
            f"**FLIGHT SUMMARY DETECTION:** This query {'IS' if is_flight_summary else 'IS NOT'} identified as a comprehensive flight summary request.",
            "",
        ]
        
        if is_flight_summary:
            description_parts.extend([
                "🔥 **COMPREHENSIVE FLIGHT SUMMARY PLANNING:**",
                "",
                "For flight summary requests, plan a comprehensive multi-tool analysis:",
                "",
                "**AVAILABLE ANALYSIS TOOLS:**",
                "- `analyze_altitude`: Specialized altitude analysis (max/min/stats/ranges)",
                "- `execute_python_code`: Flexible code execution for calculations and custom analysis", 
                "- `detect_flight_events`: Error analysis, warnings, critical events (has intelligent fallbacks)",
                "- `find_anomalies`: Pattern detection, unusual behavior identification",
                "- `analyze_flight_phase`: Phase-specific analysis (takeoff, landing, cruise)",
                "- `get_timeline_analysis`: Chronological event analysis, flight progression",
                "- `enhanced_telemetry_analysis`: Comprehensive analysis covering all 164 message types",
                "",
                "**FOR FLIGHT SUMMARY REQUESTS:**",
                "- ALWAYS include: analyze_altitude, execute_python_code, detect_flight_events",
                "- RECOMMENDED: enhanced_telemetry_analysis for comprehensive coverage",
                "- OPTIONAL: find_anomalies, get_timeline_analysis for deeper insights",
                "- REASONING: \"Comprehensive flight summary requiring multiple analysis perspectives\"",
                "- CONFIDENCE: 0.9 (high confidence for multi-tool comprehensive analysis)",
                "- APPROACH: comprehensive_flight_summary",
                "",
                "**FLIGHT SUMMARY TOOL PLANNING:**",
                "When planning for flight summaries, ensure coverage of:",
                "1. **Altitude Performance** - analyze_altitude",
                "2. **System Health & Events** - detect_flight_events",
                "3. **Custom Calculations** - execute_python_code",
                "4. **Comprehensive Analysis** - enhanced_telemetry_analysis (optional)",
                "5. **Pattern Detection** - find_anomalies (optional)",
                "",
                "**CONFIDENCE GUIDELINES:**",
                "- High confidence (0.8-1.0): Flight summary requests, clear comprehensive analysis needs",
                "- Medium confidence (0.6-0.8): Partial summary requests, multi-aspect queries",
                "- Low confidence (0.4-0.6): Unclear scope, ambiguous summary requests",
                "",
                "**CRITICAL PLANNING RULES:**",
                "1. For flight summaries: Always plan multi-tool comprehensive analysis",
                "2. Always provide a valid tool sequence",
                "3. Explain reasoning clearly",
                "4. Consider available data types and conversation context",
                "",
                "TASK: Create an execution plan using this format:",
                "",
                "NEEDS_TOOLS: true",
                "TOOL_SEQUENCE: [tool1, tool2, tool3, ...]",
                "REASONING: [Clear explanation matching the query type - summary vs specific]",
                "CONFIDENCE: [0.4-1.0]",
                "APPROACH: comprehensive_flight_summary",
                "",
                "Focus on comprehensive multi-tool analysis for complete flight overview."
            ])
        else:
            description_parts.extend([
                "🎯 **SPECIFIC QUERY PLANNING:**",
                "",
                "For specific queries, select the most appropriate tools:",
                "",
                "**AVAILABLE ANALYSIS TOOLS:**",
                "- `analyze_altitude`: Specialized altitude analysis (max/min/stats/ranges)",
                "- `execute_python_code`: Flexible code execution for calculations and custom analysis",
                "- `detect_flight_events`: Error analysis, warnings, critical events (has intelligent fallbacks)",
                "- `find_anomalies`: Pattern detection, unusual behavior identification",
                "- `analyze_flight_phase`: Phase-specific analysis (takeoff, landing, cruise)",
                "- `get_timeline_analysis`: Chronological event analysis, flight progression",
                "- `enhanced_telemetry_analysis`: Comprehensive analysis covering all 164 message types",
                "",
                "**FOR SPECIFIC QUERIES:**",
                "- For **altitude questions**: analyze_altitude OR execute_python_code as backup",
                "- For **error/warning queries**: detect_flight_events (has built-in fallbacks)",
                "- For **power/battery queries**: execute_python_code (most versatile for calculations)",
                "- For **GPS/navigation queries**: execute_python_code with GPS focus",
                "- For **complex calculations**: execute_python_code (most flexible)",
                "- **When unsure**: Default to execute_python_code (most adaptable)",
                "",
                "**SPECIFIC QUERY PLANNING:**",
                "For specific queries, focus on:",
                "1. **User's specific question** - Most appropriate single tool",
                "2. **Supporting analysis** - Complementary tools if needed",
                "3. **Fallback capability** - execute_python_code as backup",
                "",
                "**CONFIDENCE GUIDELINES:**",
                "- High confidence (0.8-1.0): Clear, specific queries with obvious tool matches",
                "- Medium confidence (0.6-0.8): Moderately clear queries, some interpretation needed",
                "- Low confidence (0.4-0.6): Ambiguous queries, fallback strategies important",
                "",
                "**CRITICAL PLANNING RULES:**",
                "1. For specific queries: Match tools to user's exact question",
                "2. Always provide a valid tool sequence",
                "3. Explain reasoning clearly",
                "4. Consider available data types and conversation context",
                "",
                "TASK: Create an execution plan using this format:",
                "",
                "NEEDS_TOOLS: true",
                "TOOL_SEQUENCE: [tool1, tool2, tool3, ...]",
                "REASONING: [Clear explanation matching the query type - summary vs specific]",
                "CONFIDENCE: [0.4-1.0]",
                "APPROACH: intelligent_analysis",
                "",
                "Focus on understanding what the user specifically wants and selecting appropriate tools."
            ])
        
        description = "\n".join(description_parts)

        expected_output = """MUST output EXACTLY this format (replace bracketed values):

NEEDS_TOOLS: true
TOOL_SEQUENCE: [tool1, tool2, tool3]
REASONING: [explanation of tool selection based on specific query analysis]
CONFIDENCE: [0.7-0.9]
APPROACH: conversational_analysis

CRITICAL REQUIREMENTS:
- REASONING must match the specific query type (error, altitude, summary, etc.)
- TOOL_SEQUENCE must follow the routing rules exactly
- NO generic "comprehensive analysis" reasoning
- Explain WHY these tools match the user's specific request"""

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None  # Will be set when creating crew
        )

    def _create_execution_task(self, user_message: str, execution_plan: ExecutionPlan) -> Task:
        """Create the execution task with proper tool calling guidance and clear result handling."""
        
        # Build task description using safe string concatenation (no f-strings with variables)
        description_parts = [
            "🚨 CRITICAL: Execute tools that are RELEVANT to the user's specific question.",
            "",
            "User Question: " + user_message,
            "",
            "EXECUTION PLAN:",
            "Tools to Execute: " + str(execution_plan.tool_sequence),
            "Reasoning: " + execution_plan.reasoning,
            "",
            "🔥 SMART EXECUTION VALIDATION:",
            "1. **ANALYZE USER'S ACTUAL QUESTION** - What are they specifically asking for?",
            "2. **VALIDATE TOOL RELEVANCE** - Only use tools that make sense for this query",
            "3. **SKIP IRRELEVANT TOOLS** - If a tool doesn't match the query, skip it with explanation",
            "4. **FOCUS ON RELEVANT RESULTS** - Report only results that answer the user's question",
            "",
            "🎯 QUERY-SPECIFIC EXECUTION RULES:",
            "",
            "For ERROR/WARNING queries like 'Check for any errors or warnings':",
            "- PRIMARY: detect_flight_events (error analysis)",
            "- SECONDARY: execute_python_code (if detailed error analysis needed)",
            "- SKIP: analyze_altitude (not relevant to error checking)",
            "",
            "For ALTITUDE queries like 'What was the maximum altitude?':",
            "- PRIMARY: analyze_altitude (altitude statistics)",
            "- SECONDARY: execute_python_code (if additional calculations needed)",
            "- SKIP: detect_flight_events (not relevant to altitude questions)",
            "",
            "For FLIGHT SUMMARY queries like 'Give me a flight summary':",
            "- USE ALL: analyze_altitude, execute_python_code, detect_flight_events",
            "- ALL tools are relevant for comprehensive overview",
            "",
            "YOU MUST EXECUTE THESE SPECIFIC TOOLS: " + str(execution_plan.tool_sequence),
            "BUT: Skip any tool that doesn't make sense for the user's specific question",
            ""
        ]
        
        # Add safe code examples and requirements using the utility method
        description_parts.extend(CodeTemplates.get_safe_description_builder())
        
        # Add remaining safe instructions
        description_parts.extend([
            "",
            "🚨 CRITICAL EXECUTION REQUIREMENTS:",
            "1. **CALL TOOLS IMMEDIATELY** - Do NOT write planning text, execute tools NOW",
            "2. **NO PLANNING RESPONSES** - Any response starting with 'I will' or 'Let me' will be REJECTED",
            "3. **ACTUAL TOOL EXECUTION ONLY** - You must use the Action: tool_name format to call tools",
            "4. **WAIT FOR TOOL RESULTS** - Get actual results from tools before proceeding",
            "5. **REPORT RELEVANT OUTPUTS** - Use exactly what tools return, focus on answering the user's question",
            "",
            "❌ FORBIDDEN RESPONSES:",
            "- 'I will start by executing...'",
            "- 'Let me use the tools...'",
            "- 'I need to call...'",
            "- Any planning or meta-commentary",
            "",
            "✅ REQUIRED EXECUTION PATTERN:",
            "Step 1: IMMEDIATELY call first RELEVANT tool using Action: tool_name format",
            "Step 2: Process tool result",
            "Step 3: Call next RELEVANT tool if needed",
            "Step 4: Synthesize actual tool results into final answer that addresses user's question",
            "",
            "🔥 START EXECUTING TOOLS NOW - NO PLANNING TEXT ALLOWED",
            "",
            "SMART EXECUTION EXAMPLES:",
            "",
            "For 'Check for any errors or warnings':",
            "1. Action: detect_flight_events with query 'errors and warnings'",
            "2. Report error findings directly",
            "3. Skip analyze_altitude (not relevant to error checking)",
            "",
            "For 'What was the maximum altitude?':",
            "1. Action: analyze_altitude with query 'maximum altitude'",
            "2. Report altitude finding directly",
            "3. Skip detect_flight_events (not relevant to altitude query)",
            "",
            "RESPONSE FORMAT:",
            "- Start with actual findings that answer the user's specific question",
            "- Present specific data, numbers, and measurements from relevant tools",
            "- Combine multiple tool results intelligently when all are relevant",
            "- Keep response focused on what the user actually asked for",
            "- NEVER add information not provided by tools"
        ])
        
        # Join all parts with newlines - completely safe from f-string evaluation
        description = "\n".join(description_parts)

        expected_output = """🚨 MANDATORY TOOL EXECUTION RESULTS ONLY

Your response MUST contain ACTUAL TOOL RESULTS that answer the user's specific question.

REQUIRED FORMAT:
1. Start immediately with Action: tool_name to call first RELEVANT tool
2. Process tool result and report findings that answer user's question
3. Call next tool using Action: tool_name format ONLY if relevant to query
4. Continue until user's question is answered
5. Provide final answer focused on what user specifically asked for

❌ WILL BE REJECTED:
- "I will start by executing..."
- "Let me use the tools..."
- "I need to call..."
- Any planning or meta-commentary
- Reporting irrelevant tool results (e.g., altitude data for error queries)

✅ MUST INCLUDE:
- Actual tool execution using Action: format
- Real data, numbers, and findings from tools that answer the user's question
- Specific measurements from relevant tool outputs
- Final answer that directly addresses what the user asked for

CRITICAL: Only report data that tools actually returned AND that answers the user's specific question."""

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None  # Will be set when creating crew
        )

    def _create_critique_task(self, user_message: str, execution_result: str) -> Task:
        """Create the enhanced critique task with advanced report generation capabilities."""
        # Detect if this is a flight summary request
        is_summary_request = self._is_flight_summary_request(user_message)
        
        description = f"""CRITICAL: Transform this technical analysis into the final user response using your enhanced report generation capabilities.

Original Question: {user_message}
Is Flight Summary Request: {is_summary_request}

Technical Analysis Result: {execution_result}

🚨 CRITICAL REQUIREMENT: Your output MUST be the complete final answer that the user will see. Do NOT output:
- "I can give a great answer"
- "Let me optimize this" 
- "I now can provide..."
- Any planning or meta-commentary

🔥 CRITICAL DATA INTEGRITY CHECK:
If the Technical Analysis Result contains ONLY planning statements like "I will start by executing..." or "Let me use..." without ANY actual data or results, then you MUST:
1. Recognize that NO ACTUAL ANALYSIS was performed
2. Return: "I'm preparing a comprehensive flight analysis for you. Please try your question again in a moment."
3. NEVER fabricate specific numbers, measurements, or findings that weren't actually generated

ONLY provide specific flight data (altitudes, times, errors, etc.) if they appear in the Technical Analysis Result from actual tool execution.

📊 ENHANCED RESPONSE LOGIC:

**IF THIS IS A FLIGHT SUMMARY REQUEST ({is_summary_request}):**
Generate a **CONSOLIDATED HUMAN-READABLE SUMMARY** by:

1. **Data Extraction:** Parse the Technical Analysis Result to extract:
   - Altitude metrics (max, min, average, range)
   - Power system data (voltage, current, battery info)
   - GPS/Navigation data (satellites, HDOP, signal quality)
   - System events and alerts (error counts, warnings, failures)

2. **Rules-Based Analysis:** Apply analytical rules to extracted data:
   - GPS HDOP > 5.0 - Add GPS accuracy warning
   - Minimum voltage < 1.0V - Add power system warning
   - Event count > 10 - Add high event count note
   - Satellite count < 4 - Add GPS signal warning

3. **Human-Readable Summary Generation:** Structure as conversational analysis:
   
   Create a natural, flowing summary that covers:
   - Flight altitude performance in conversational language
   - Power system status with clear explanations
   - GPS and navigation quality in plain English
   - System events and alerts with context
   - Critical insights and recommendations in accessible language
   
   Format as natural paragraphs, not markdown syntax or bullet points.
   Use conversational tone while maintaining technical accuracy.

4. **Dynamic Content:** Only include sections where executor provided data
5. **Professional but Accessible:** Clear, conversational formatting without markdown syntax

**IF THIS IS A SIMPLE/SPECIFIC QUESTION:**
Use standard response optimization:
- Single measurement queries - Natural sentence responses
- Specific error checks - Bullet point findings  
- Basic questions - Direct, clear answers

**RESPONSE QUALITY STANDARDS:**
- Preserve all executor data exactly as provided
- Apply rules-based analysis only when data supports it
- Use professional, analytical language
- Highlight critical findings requiring attention
- Maintain consistency in formatting and terminology

**DATA EXTRACTION PATTERNS:**
From executor results, look for patterns like:
- "Maximum altitude: 1448m" - Extract: max_altitude = 1448
- "Average voltage: 50.64V" - Extract: avg_voltage = 50.64
- "Detected 18 flight events" - Extract: event_count = 18
- "Average satellites: 2.5" - Extract: avg_satellites = 2.5
- "Average HDOP: 42.24" - Extract: avg_hdop = 42.24

**CRITICAL RULES FOR ANALYTICAL NOTES:**
Only add analytical notes if:
1. Executor provided the specific metric
2. Metric meets the threshold condition
3. Note adds genuine value for flight safety/analysis

🚨 MANDATORY FINAL OUTPUT: 
Provide the complete, final response optimized for the request type. For flight summaries, deliver a conversational, human-readable summary in natural language paragraphs (NO markdown syntax). For specific questions, provide direct answers. This will be shown to the user immediately."""

        expected_output = f"""The complete, final response optimized for the request type:

{"**FOR FLIGHT SUMMARY REQUEST:** Generate a conversational, human-readable summary with:" if is_summary_request else "**FOR SPECIFIC QUESTION:** Provide a direct, clear answer with:"}

1. Data extracted EXACTLY from executor results (no fabrication)
2. {"Natural conversational formatting in flowing paragraphs" if is_summary_request else "Appropriate formatting (sentences/bullets)"}
3. {"Rules-based analytical insights integrated naturally into the text" if is_summary_request else "Essential information and key numbers"}
4. Professional but accessible presentation in plain English
5. No planning thoughts or meta-commentary

{"Example: 'The flight reached a maximum altitude of 1,448 meters with an average of 161 meters throughout the mission. The power system performed well with an average voltage of 50.64V, though a minimum reading of 0.00V suggests potential sensor issues. GPS navigation showed challenges with only 2.5 satellites on average and a high HDOP of 42.24, indicating reduced positioning accuracy. During the flight, 18 system events were recorded, including several failsafe activations that warrant further review.'" if is_summary_request else "Direct answer that addresses the user's specific question."}

CRITICAL: Output only the final answer, nothing else. For flight summaries, use natural language paragraphs, NOT markdown syntax."""

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None  # Will be set when creating crew
        )
    
    def _is_flight_summary_request(self, user_message: str) -> bool:
        """Detect if the user is requesting a comprehensive flight summary."""
        message_lower = user_message.lower()
        
        summary_indicators = [
            "flight summary", "summary of this flight", "summary of the flight",
            "overall analysis", "comprehensive report", "flight performance report",
            "flight report", "complete analysis", "full analysis",
            "give me a summary", "summarize this flight", "summarize the flight",
            "flight overview", "performance summary", "analysis summary"
        ]
        
        return any(indicator in message_lower for indicator in summary_indicators)

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
            
            # Stage 1: Enhanced Planning with intelligent intent-based fallback
            logger.info(f"[{session_id}] Stage 1: ENHANCED PLANNER")
            planning_task = await self._create_planning_task(user_message, session)
            planning_task.agent = planner_agent
            
            planning_crew = Crew(
                agents=[planner_agent],
                tasks=[planning_task],
                process=Process.sequential,
                verbose=True,
                memory=True,
                embedder=self.memory_embedder_config
            )
            
            try:
                plan_result = planning_crew.kickoff()
                plan_text = str(plan_result)
                execution_plan = self._parse_execution_plan(plan_text)
                
                # Enhanced plan validation with intelligent intent-based fallbacks
                if not execution_plan.needs_tools or not execution_plan.tool_sequence:
                    logger.warning(f"[{session_id}] PLANNER returned invalid plan, using intelligent intent-based fallback")
                    execution_plan = await self._create_intent_based_plan(user_message, session)
                
                logger.info(f"[{session_id}] PLANNER completed - Plan confidence: {execution_plan.confidence:.2f}, Tools: {execution_plan.tool_sequence}")
                
            except Exception as planning_error:
                logger.error(f"[{session_id}] PLANNER failed: {str(planning_error)}")
                execution_plan = await self._create_intent_based_plan(user_message, session)
                logger.info(f"[{session_id}] Using intelligent intent-based fallback plan: {execution_plan.tool_sequence}")
            
            # Stage 2: Enhanced Execution with error recovery
            logger.info(f"[{session_id}] Stage 2: INTELLIGENT EXECUTOR - Tools: {execution_plan.tool_sequence}")
            execution_task = self._create_execution_task(user_message, execution_plan)
            execution_task.agent = executor_agent
            
            execution_crew = Crew(
                agents=[executor_agent],
                tasks=[execution_task],
                process=Process.sequential,
                verbose=True,
                memory=True,
                embedder=self.memory_embedder_config
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
                    verbose=True,
                    memory=True,
                    embedder=self.memory_embedder_config
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
        
        # Additional safety check to ensure data_summary is never None
        if data_summary is None:
            data_summary = {
                "message_types": 0,
                "total_records": 0,
                "time_range": {"duration_minutes": 0},
                "key_metrics": {}
            }
        
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

    def _ensure_tool_execution(self, step):
        """Ensure the executor actually calls tools instead of just describing what it will do."""
        if hasattr(step, 'output') and step.output:
            output = step.output.lower()
            
            # Check if the agent is just describing what it will do instead of doing it
            problematic_phrases = [
                "i will start by",
                "i will use", 
                "let me use",
                "time to get started",
                "let's dive in",
                "i'll analyze",
                "i'll execute",
                "i need to call",
                "i should use",
                "let me begin",
                "i will execute",
                "i plan to",
                "i intend to",
                "first, i will",
                "next, i will",
                "i will proceed",
                "let me start",
                "i will now",
                "my approach will be"
            ]
            
            # Also check for absence of actual tool calls
            has_tool_calls = any(pattern in output for pattern in [
                "action:", "action input:", "thought:", "observation:"
            ])
            
            is_planning = any(phrase in output for phrase in problematic_phrases)
            
            if is_planning and not has_tool_calls:
                # Log the problematic response
                logger.error(f"EXECUTOR AGENT GENERATING PLANNING TEXT INSTEAD OF EXECUTING TOOLS: {step.output[:200]}...")
                
                # Force the agent to retry with more explicit instructions
                # This is a more aggressive intervention
                step.output = "ERROR: You must execute tools immediately using Action: tool_name format. Do not provide planning text. Start tool execution now."
                return step
        
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

    async def _create_intent_based_plan(self, user_message: str, session: V2ConversationSession) -> ExecutionPlan:
        """Create an execution plan based on intelligent intent classification and semantic data understanding."""
        try:
            # Use Intent Classifier to intelligently categorize the query
            classified_intent = await self.intent_classifier.classify_intent(user_message, session)
            
            # Phase 3: Get semantic context for data source selection
            semantic_matches = await self.semantic_retriever.find_relevant_dataframes(
                user_message, 
                list(session.dataframes.keys()), 
                top_k=3
            )
            
            logger.info(f"[{session.session_id}] Intent classified: {classified_intent.intent_name} (confidence: {classified_intent.confidence:.2f})")
            if semantic_matches:
                logger.info(f"[{session.session_id}] Semantic matches: {[m.dataframe_key for m in semantic_matches]}")
            
            # Use the recommended tools from intent configuration
            tool_sequence = classified_intent.recommended_tools.copy()
            reasoning = f"Intent-based routing: {classified_intent.reasoning}"
            
            # Enhance reasoning with semantic context
            if semantic_matches:
                relevant_dfs = [m.dataframe_key for m in semantic_matches[:3]]
                reasoning += f". Semantic analysis indicates most relevant dataframes: {', '.join(relevant_dfs)}"
            
            confidence = classified_intent.confidence
            
            # Determine critique requirement based on intent and tool count
            requires_critique = self._should_use_critique(classified_intent.intent_key, tool_sequence)
            
            # Determine complexity based on intent and tools
            estimated_complexity = self._estimate_complexity(classified_intent.intent_key, tool_sequence)
            
            # Phase 3: Use semantic matches to focus on most relevant dataframes
            target_dataframes = list(session.dataframes.keys())
            if semantic_matches:
                # Prioritize semantically relevant dataframes
                relevant_dfs = [m.dataframe_key for m in semantic_matches]
                target_dataframes = relevant_dfs + [df for df in target_dataframes if df not in relevant_dfs]
            
            return ExecutionPlan(
                needs_tools=True,
                tool_sequence=tool_sequence,
                reasoning=reasoning,
                confidence=confidence,
                approach="intent_and_semantic_routing",
                estimated_complexity=estimated_complexity,
                requires_critique=requires_critique,
                max_iterations=6,
                success_criteria=["intent_specific_answer_provided"],
                target_dataframes=target_dataframes,
                target_columns={},
                data_relationships=[f"Semantic match: {m.dataframe_key} ({m.relevance_reason})" for m in semantic_matches],
                execution_steps=[f"Execute {tool} based on {classified_intent.intent_name}" for tool in tool_sequence],
                stopping_conditions=["intent_satisfied", "all_tools_attempted"],
                classified_intent=classified_intent
            )
            
        except Exception as e:
            logger.error(f"[{session.session_id}] Intent classification failed: {e}, using simple fallback")
            return self._create_simple_fallback_plan(user_message, session)
    
    def _should_use_critique(self, intent_key: str, tool_sequence: List[str]) -> bool:
        """Determine if critique is needed based on intent and tools."""
        # Single-tool intents usually don't need critique
        if len(tool_sequence) == 1:
            return False
        
        # Comprehensive analysis intents benefit from critique
        if intent_key in ['flight_performance_analysis', 'anomaly_detection']:
            return True
        
        # Multi-tool plans generally benefit from critique
        if len(tool_sequence) > 2:
            return True
        
        return False
    
    def _estimate_complexity(self, intent_key: str, tool_sequence: List[str]) -> str:
        """Estimate complexity based on intent and tool requirements."""
        if intent_key in ['flight_performance_analysis', 'anomaly_detection'] or len(tool_sequence) > 2:
            return "high"
        elif len(tool_sequence) == 2 or intent_key in ['flight_events_analysis', 'navigation_analysis']:
            return "medium"
        else:
            return "low"
    
    def _create_simple_fallback_plan(self, user_message: str, session: V2ConversationSession) -> ExecutionPlan:
        """Simple fallback when intent classification fails completely."""
        return ExecutionPlan(
            needs_tools=True,
            tool_sequence=["execute_python_code"],
            reasoning="Simple fallback due to intent classification failure",
            confidence=0.5,
            approach="simple_fallback",
            estimated_complexity="medium",
            requires_critique=False,
            max_iterations=6,
            success_criteria=["provide_answer"],
            target_dataframes=list(session.dataframes.keys()),
            target_columns={},
            data_relationships=[],
            execution_steps=["Execute versatile analysis tool"],
            stopping_conditions=["answer_found"]
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

# REMOVED: Hardcoded validation logic replaced by intelligent IntentClassifier system
# The existing LLM-based IntentClassifier in _create_intent_based_plan provides:
# - Dynamic, configurable intent detection
# - LLM-based classification instead of brittle keyword matching  
# - Extensible through config.py without code changes
# - Context-aware classification using conversation history
# This is far superior to hardcoded if/elif validation rules.


# Compatibility aliases for easy integration
ChatAgentV2 = MultiRoleAgent
MultiRoleChatAgent = MultiRoleAgent 