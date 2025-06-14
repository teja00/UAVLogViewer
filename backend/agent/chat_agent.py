"""
Multi-role LLM chat agent for UAV log analysis.
Implements a stack of cooperating LLM roles: Planner → Executor → Critic
"""

import logging
import uuid
import json
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass

from openai import OpenAI
from config import get_settings
from models import ChatMessage, V2ConversationSession

from utils.documentation import DocumentationService
from utils.log_parser import LogParserService, get_data_summary
from tools.tool_definitions import get_tool_definitions
from tools.analysis_tools import AnalysisTools

logger = logging.getLogger(__name__)


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


@dataclass
class ExecutionPlan:
    """Structured plan from the Planner role."""
    needs_tools: bool
    tool_sequence: List[str]
    reasoning: str
    confidence: float
    approach: str


class MultiRoleChatAgent:
    """Multi-role LLM chat agent with specialized roles for planning, execution, and critique."""

    def __init__(self):
        """Initialize the multi-role chat agent with specialized LLM instances."""
        self.settings = get_settings()
        
        # Initialize role-specific OpenAI clients
        if self.settings.openai_api_key:
            # Planner: High-precision, low temperature for consistent planning
            self.planner_llm = OpenAI(api_key=self.settings.openai_api_key)
            
            # Executor: Moderate temperature for tool calling and analysis
            self.executor_llm = OpenAI(api_key=self.settings.openai_api_key)
            
            # Critic: Low temperature for consistent quality assessment
            self.critic_llm = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.planner_llm = None
            self.executor_llm = None
            self.critic_llm = None
            logger.warning("OpenAI API key not configured.")
        
        # Initialize services
        self.documentation_service = DocumentationService()
        self.log_parser = LogParserService()
        self.analysis_tools = AnalysisTools()
        
        # Session storage and call tracking
        self.sessions: Dict[str, V2ConversationSession] = {}
        self.llm_calls: List[LLMCall] = []

    async def create_or_get_session(self, session_id: Optional[str] = None) -> V2ConversationSession:
        """Creates a new session or retrieves an existing one."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = V2ConversationSession(session_id=session_id)
            logger.info(f"Created new chat session: {session_id}")
        
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

    async def chat(self, session_id: str, user_message: str) -> str:
        """Multi-role chat method: Planner → Executor → Critic."""
        logger.info(f"[{session_id}] Multi-role chat request: '{user_message[:50]}...'")
        
        if not all([self.planner_llm, self.executor_llm, self.critic_llm]):
            return "AI service unavailable. Check configuration."

        session = await self.create_or_get_session(session_id)

        if session.is_processing:
            return "Still processing log. Please wait."

        if session.processing_error:
            return f"Log processing error: {session.processing_error}"

        if not session.dataframes:
            return "No flight data loaded. Please upload a log file first."

        session.messages.append(ChatMessage(role="user", content=user_message))

        try:
            # Step 1: Planner - Analyze request and create execution plan
            logger.info(f"[{session_id}] PLANNER: Analyzing user request and creating plan")
            execution_plan = await self._planner_role(session, user_message)
            
            # Step 2: Executor - Follow plan and execute tools
            logger.info(f"[{session_id}] EXECUTOR: Following plan and executing analysis")
            draft_answer = await self._executor_role(session, user_message, execution_plan)
            
            # Step 3: Critic - Review and refine the answer
            logger.info(f"[{session_id}] CRITIC: Reviewing and refining answer")
            final_answer = await self._critic_role(session, user_message, draft_answer, execution_plan)
            
            session.messages.append(ChatMessage(role="assistant", content=final_answer))
            
            # Log role performance summary
            self._log_role_performance(session_id)
            
            return final_answer

        except Exception as e:
            logger.error(f"Error in multi-role chat processing: {e}", exc_info=True)
            return "Unexpected error in analysis. Please try again."

    async def _planner_role(self, session: V2ConversationSession, user_message: str) -> ExecutionPlan:
        """Planner role: Analyzes user request and creates structured execution plan."""
        data_summary = get_data_summary(session)
        available_tools = get_tool_definitions()
        
        planner_prompt = self._get_planner_prompt(data_summary, available_tools)
        
        start_time = datetime.now()
        
        try:
            response = self.planner_llm.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": planner_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,  # Low temperature for consistent planning
                max_tokens=800,
                timeout=20
            )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            # Track this LLM call
            self._track_llm_call("planner", response, duration)
            
            # Parse the structured plan from the response
            plan_text = response.choices[0].message.content
            return self._parse_execution_plan(plan_text)
            
        except Exception as e:
            logger.error(f"Planner role error: {e}")
            # Return fallback plan
            return ExecutionPlan(
                needs_tools=True,
                tool_sequence=["execute_python_code"],
                reasoning="Fallback plan due to planner error",
                confidence=0.5,
                approach="direct_analysis"
            )

    async def _executor_role(self, session: V2ConversationSession, user_message: str, plan: ExecutionPlan) -> str:
        """Executor role: Follows the plan and executes tools to generate analysis."""
        data_summary = get_data_summary(session)
        executor_prompt = self._get_executor_prompt(data_summary, plan)
        
        start_time = datetime.now()
        
        try:
            if not plan.needs_tools:
                # Direct response without tools
                response = self.executor_llm.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": executor_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.2,  # Moderate temperature for analysis
                    max_tokens=1000,
                    timeout=25
                )
                
                duration = (datetime.now() - start_time).total_seconds() * 1000
                self._track_llm_call("executor", response, duration)
                
                return response.choices[0].message.content or "Unable to provide analysis."
            
            else:
                # Execute with tools according to plan
                tools = get_tool_definitions()
                
                response = self.executor_llm.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": executor_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3,  # Moderate temperature for tool execution
                    timeout=30
                )
                
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                
                if not tool_calls:
                    duration = (datetime.now() - start_time).total_seconds() * 1000
                    self._track_llm_call("executor", response, duration)
                    return response_message.content or "No analysis performed."
                
                # Process tool calls
                tool_results = []
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"[{session.session_id}] EXECUTOR: Executing tool {function_name}")
                    result = await self._execute_tool(function_name, function_args, session)
                    
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result),
                    })
                
                # Get final analysis from executor
                final_response = self.executor_llm.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": executor_prompt},
                        {"role": "user", "content": user_message},
                        response_message,
                    ] + tool_results,
                    temperature=0.3,
                    timeout=30
                )
                
                duration = (datetime.now() - start_time).total_seconds() * 1000
                self._track_llm_call("executor", final_response, duration)
                
                return final_response.choices[0].message.content or "Analysis completed but no result generated."
                
        except Exception as e:
            logger.error(f"Executor role error: {e}")
            return f"Analysis error: Unable to complete execution. {str(e)[:100]}"

    async def _critic_role(self, session: V2ConversationSession, user_message: str, 
                          draft_answer: str, plan: ExecutionPlan) -> str:
        """Critic role: Reviews draft answer for quality, accuracy, and completeness."""
        data_summary = get_data_summary(session)
        critic_prompt = self._get_critic_prompt(data_summary, plan)
        
        start_time = datetime.now()
        
        try:
            response = self.critic_llm.chat.completions.create(
                model="gpt-3.5-turbo-16k",  # Use cheaper model for critique
                messages=[
                    {"role": "system", "content": critic_prompt},
                    {"role": "user", "content": f"""USER QUESTION: {user_message}

EXECUTION PLAN:
- Needs tools: {plan.needs_tools}
- Approach: {plan.approach}
- Reasoning: {plan.reasoning}

DRAFT ANSWER:
{draft_answer}

Please review this answer and provide the final refined version."""}
                ],
                temperature=0.0,  # Very low temperature for consistent critique
                max_tokens=1200,
                timeout=20
            )
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_llm_call("critic", response, duration)
            
            refined_answer = response.choices[0].message.content
            
            # If critic suggests significant changes, return refined version
            # Otherwise return original draft
            if refined_answer and len(refined_answer) > len(draft_answer) * 0.5:
                return refined_answer
            else:
                return draft_answer
                
        except Exception as e:
            logger.error(f"Critic role error: {e}")
            # Return draft answer if critic fails
            return draft_answer

    def _get_planner_prompt(self, data_summary: Dict[str, Any], available_tools: List[Dict]) -> str:
        """Generate specialized prompt for the Planner role."""
        tool_names = [tool["function"]["name"] for tool in available_tools]
        
        return f"""You are the PLANNER role in a multi-agent UAV flight data analysis system.

Your job is to analyze user requests and create structured execution plans.

AVAILABLE DATA:
- Message types: {data_summary.get('message_types', 0)}
- Total records: {data_summary.get('total_records', 0)}
- Time range: {data_summary.get('time_range', 'Unknown')}

AVAILABLE ANALYSIS TOOLS:
{', '.join(tool_names)}

For each user request, provide a structured plan in this exact format:

NEEDS_TOOLS: [true/false]
TOOL_SEQUENCE: [list of tools to use in order]
REASONING: [why this approach is best]
CONFIDENCE: [0.0-1.0 confidence in this plan]
APPROACH: [direct_response/tool_analysis/complex_multi_step]

Guidelines:
- Use tools for calculations, data analysis, and complex queries
- Prefer simple direct responses for basic questions
- Consider tool dependencies and logical sequence
- Be concise but thorough in reasoning"""

    def _get_executor_prompt(self, data_summary: Dict[str, Any], plan: ExecutionPlan) -> str:
        """Generate specialized prompt for the Executor role."""
        return f"""You are the EXECUTOR role in a multi-agent UAV flight data analysis system.

Your job is to follow the execution plan and provide technical analysis.

EXECUTION PLAN:
- Needs tools: {plan.needs_tools}
- Tool sequence: {plan.tool_sequence}
- Approach: {plan.approach}
- Reasoning: {plan.reasoning}

AVAILABLE DATA:
- Message types: {data_summary.get('message_types', 0)}
- Total records: {data_summary.get('total_records', 0)}

Instructions:
- Follow the plan created by the Planner
- Execute tools in the recommended sequence
- Provide technical, accurate analysis
- Include specific numbers and findings
- Focus on data-driven insights
- Be thorough but concise"""

    def _get_critic_prompt(self, data_summary: Dict[str, Any], plan: ExecutionPlan) -> str:
        """Generate specialized prompt for the Critic role."""
        return f"""You are the CRITIC role in a multi-agent UAV flight data analysis system.

Your job is to review draft answers for quality, accuracy, and user-friendliness.

Review the draft answer and check for:
1. ACCURACY: Are the technical details correct?
2. COMPLETENESS: Does it fully answer the user's question?
3. CLARITY: Is it understandable to the user?
4. HALLUCINATIONS: Any made-up data or false claims?
5. RELEVANCE: Does it address what was actually asked?

If the draft answer needs improvement, provide a refined version that:
- Keeps all accurate technical information
- Makes the language more user-friendly
- Removes any hallucinations or speculation
- Adds missing context if needed
- Maintains a helpful, professional tone

If the draft answer is already good, you may return it as-is or with minor improvements."""

    def _parse_execution_plan(self, plan_text: str) -> ExecutionPlan:
        """Parse structured execution plan from planner response."""
        try:
            lines = plan_text.strip().split('\n')
            
            needs_tools = False
            tool_sequence = []
            reasoning = ""
            confidence = 0.7
            approach = "direct_response"
            
            for line in lines:
                line = line.strip()
                if line.startswith("NEEDS_TOOLS:"):
                    needs_tools = "true" in line.lower()
                elif line.startswith("TOOL_SEQUENCE:"):
                    # Extract tools from list format
                    tools_str = line.split(":", 1)[1].strip()
                    if "[" in tools_str and "]" in tools_str:
                        tools_str = tools_str.strip("[]")
                        tool_sequence = [t.strip().strip("'\"") for t in tools_str.split(",") if t.strip()]
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        confidence = 0.7
                elif line.startswith("APPROACH:"):
                    approach = line.split(":", 1)[1].strip()
            
            return ExecutionPlan(
                needs_tools=needs_tools,
                tool_sequence=tool_sequence,
                reasoning=reasoning,
                confidence=confidence,
                approach=approach
            )
            
        except Exception as e:
            logger.error(f"Error parsing execution plan: {e}")
            return ExecutionPlan(
                needs_tools=True,
                tool_sequence=["execute_python_code"],
                reasoning="Fallback plan due to parsing error",
                confidence=0.5,
                approach="tool_analysis"
            )

    def _track_llm_call(self, role: str, response: Any, duration_ms: float):
        """Track LLM call for cost and performance monitoring."""
        try:
            # Extract token usage if available
            usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
            completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
            
            # Rough cost estimation (varies by model)
            cost_per_1k_prompt = 0.03 if "gpt-4" in self.settings.openai_model else 0.001
            cost_per_1k_completion = 0.06 if "gpt-4" in self.settings.openai_model else 0.002
            
            cost_estimate = (prompt_tokens * cost_per_1k_prompt / 1000) + \
                          (completion_tokens * cost_per_1k_completion / 1000)
            
            call = LLMCall(
                role=role,
                model=self.settings.openai_model,
                tokens_prompt=prompt_tokens,
                tokens_completion=completion_tokens,
                cost_estimate=cost_estimate,
                duration_ms=int(duration_ms),
                timestamp=datetime.now()
            )
            
            self.llm_calls.append(call)
            
            logger.info(f"[{role.upper()}] Tokens: {prompt_tokens}+{completion_tokens}, "
                       f"Cost: ${cost_estimate:.4f}, Duration: {duration_ms:.0f}ms")
            
        except Exception as e:
            logger.warning(f"Error tracking LLM call: {e}")

    def _log_role_performance(self, session_id: str):
        """Log performance summary for this chat session."""
        if not self.llm_calls:
            return
            
        recent_calls = [call for call in self.llm_calls if 
                       (datetime.now() - call.timestamp).total_seconds() < 60]
        
        if not recent_calls:
            return
            
        total_cost = sum(call.cost_estimate for call in recent_calls)
        total_tokens = sum(call.tokens_prompt + call.tokens_completion for call in recent_calls)
        total_duration = sum(call.duration_ms for call in recent_calls)
        
        role_summary = {}
        for call in recent_calls:
            if call.role not in role_summary:
                role_summary[call.role] = {"calls": 0, "tokens": 0, "cost": 0.0, "duration": 0}
            role_summary[call.role]["calls"] += 1
            role_summary[call.role]["tokens"] += call.tokens_prompt + call.tokens_completion
            role_summary[call.role]["cost"] += call.cost_estimate
            role_summary[call.role]["duration"] += call.duration_ms
        
        logger.info(f"[{session_id}] MULTI-ROLE PERFORMANCE SUMMARY:")
        logger.info(f"  Total: {len(recent_calls)} calls, {total_tokens} tokens, "
                   f"${total_cost:.4f}, {total_duration:.0f}ms")
        
        for role, stats in role_summary.items():
            logger.info(f"  {role.upper()}: {stats['calls']} calls, {stats['tokens']} tokens, "
                       f"${stats['cost']:.4f}, {stats['duration']:.0f}ms")

    async def _execute_tool(self, function_name: str, function_args: dict, session: V2ConversationSession) -> str:
        """Execute a tool function and return the result."""
        try:
            if function_name == "execute_python_code":
                return self.analysis_tools.execute_python_code(session, function_args["code"])
            elif function_name == "find_anomalies":
                return self.analysis_tools.find_anomalies(session, function_args["focus_areas"])
            elif function_name == "compare_metrics":
                return self.analysis_tools.compare_metrics(session, function_args["metrics"], function_args["comparison_type"])
            elif function_name == "generate_insights":
                return self.analysis_tools.generate_insights(session, function_args["focus"])
            elif function_name == "detect_flight_events":
                return self.analysis_tools.detect_flight_events(session, function_args["event_types"])
            elif function_name == "analyze_flight_phase":
                return self.analysis_tools.analyze_flight_phase(session, function_args["phase"], function_args.get("metrics", []))
            elif function_name == "get_timeline_analysis":
                return self.analysis_tools.get_timeline_analysis(session, function_args["time_resolution"])
            else:
                return f"Unknown tool: {function_name}"
        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {e}")
            return f"Tool execution error: {str(e)}"


# Backward compatibility alias
ChatAgentV2 = MultiRoleChatAgent 