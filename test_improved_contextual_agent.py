#!/usr/bin/env python3
"""
Test script for improved contextual understanding in MultiRoleAgent.
This tests the specific improvements made to handle contextual queries like
"consistently high" in UAV flight analysis.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.agent.multi_role_agent import MultiRoleAgent
from models import ChatMessage, V2ConversationSession
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def create_sample_flight_data():
    """Create sample flight data for testing contextual queries."""
    # Generate 200 data points over 4 minutes
    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(200)]
    
    # GPS data with consistent high altitude (1400-1450m range)
    gps_data = pd.DataFrame({
        'timestamp': timestamps,
        'Alt': np.random.normal(1425, 8, 200),  # Mean 1425m, std dev 8m (very stable)
        'Lat': np.random.normal(-35.123456, 0.00001, 200),
        'Lng': np.random.normal(149.123456, 0.00001, 200),
        'Status': [3] * 200,  # All 3D fixes
        'NSats': np.random.randint(8, 12, 200)  # Good satellite count
    })
    
    # BARO data (barometric altitude) - should be similar to GPS
    baro_data = pd.DataFrame({
        'timestamp': timestamps,
        'Alt': np.random.normal(1420, 6, 200),  # Slightly different but stable
        'Press': np.random.normal(1013.25, 5, 200),
        'Temp': np.random.normal(20, 2, 200)
    })
    
    # CTUN data (control tuning) showing altitude hold behavior
    ctun_data = pd.DataFrame({
        'timestamp': timestamps,
        'Alt': np.random.normal(1425, 5, 200),  # Actual altitude
        'DAlt': [1425] * 200,  # Desired altitude (constant - altitude hold)
        'ClimbRate': np.random.normal(0, 0.5, 200),  # Minimal climb rate
        'ThO': np.random.normal(0.4, 0.1, 200)  # Throttle output
    })
    
    # MODE data showing stable flight mode
    mode_data = pd.DataFrame({
        'timestamp': [timestamps[0], timestamps[50], timestamps[180]],
        'Mode': [4, 9, 4],  # Guided -> Loiter -> Guided (minimal changes)
        'asText': ['GUIDED', 'LOITER', 'GUIDED'],
        'ModeNum': [4, 9, 4]
    })
    
    # ATT data (attitude) showing stable flight
    att_data = pd.DataFrame({
        'timestamp': timestamps,
        'Roll': np.random.normal(0, 1.5, 200),  # Very stable roll
        'Pitch': np.random.normal(2, 1.2, 200),  # Slight forward pitch, stable
        'Yaw': np.random.normal(90, 3, 200)  # Stable heading
    })
    
    # CURR data (power system)
    curr_data = pd.DataFrame({
        'timestamp': timestamps,
        'Volt': np.random.normal(15.2, 0.3, 200),  # Healthy battery voltage
        'Curr': np.random.normal(12.5, 2, 200),  # Current consumption
        'CurrTot': np.cumsum(np.random.normal(0.05, 0.01, 200))  # Total consumed
    })
    
    # VIBE data (vibration)
    vibe_data = pd.DataFrame({
        'timestamp': timestamps,
        'VibeX': np.random.normal(20, 5, 200),  # Low vibration
        'VibeY': np.random.normal(22, 5, 200),
        'VibeZ': np.random.normal(18, 4, 200)
    })
    
    # MSG data with some informational messages
    msg_data = pd.DataFrame({
        'timestamp': [timestamps[10], timestamps[50], timestamps[100]],
        'Message': [
            'Flight mode changed to LOITER',
            'GPS: 3D Fix achieved',
            'Mission complete'
        ]
    })
    
    return {
        'GPS': gps_data,
        'BARO': baro_data,
        'CTUN': ctun_data,
        'MODE': mode_data,
        'ATT': att_data,
        'CURR': curr_data,
        'VIBE': vibe_data,
        'MSG': msg_data
    }

async def test_contextual_altitude_understanding():
    """Test the agent's improved understanding of contextual altitude queries."""
    print("🚀 Testing Contextual Altitude Understanding...")
    
    agent = MultiRoleAgent()
    session = await agent.create_or_get_session("test_contextual")
    
    # Load sample flight data
    sample_data = create_sample_flight_data()
    session.dataframes = sample_data
    
    # Set up conversation context (simulate previous GPS quality discussion)
    session.messages = [
        ChatMessage(role="user", content="Show GPS data quality"),
        ChatMessage(role="assistant", content="The GPS data quality for the UAV flight was consistently high, ensuring reliable and accurate positioning information throughout the entire operation.")
    ]
    
    print("📊 Sample Data Created:")
    print(f"  - GPS altitude: Mean {sample_data['GPS']['Alt'].mean():.1f}m, Std {sample_data['GPS']['Alt'].std():.1f}m")
    print(f"  - Flight duration: {len(sample_data['GPS'])} data points")
    print(f"  - Mode changes: {len(sample_data['MODE'])} transitions")
    
    # Test the specific problematic query
    query = "can you describe more in detail on why the UAV flight was consistently high?"
    print(f"\n{'='*60}")
    print(f"Test Query: {query}")
    print('='*60)
    
    try:
        response = await agent.chat(session.session_id, query)
        print(f"✅ Response ({len(response)} chars):")
        print(response)
        
        # Verify the response addresses altitude, not GPS quality
        response_lower = response.lower()
        altitude_keywords = ['altitude', 'meters', 'stability', 'consistent', 'control', 'standard deviation', 'mean']
        gps_confusion_keywords = ['gps quality', 'positioning', 'signal reception']
        
        altitude_score = sum(1 for keyword in altitude_keywords if keyword in response_lower)
        confusion_score = sum(1 for keyword in gps_confusion_keywords if keyword in response_lower)
        
        print(f"\n📈 Analysis:")
        print(f"  - Altitude-related content: {altitude_score}/{len(altitude_keywords)} keywords found")
        print(f"  - GPS confusion indicators: {confusion_score} found")
        
        if altitude_score >= 4 and confusion_score == 0:
            print("  ✅ EXCELLENT: Correctly interpreted as altitude query with detailed analysis")
        elif altitude_score >= 2 and confusion_score == 0:
            print("  ✅ PASS: Correctly interpreted as altitude query")
        elif altitude_score >= 2:
            print("  ⚠️  PARTIAL: Some altitude content, but could be better")
        else:
            print("  ❌ FAIL: Did not properly address altitude")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return session

async def test_tool_execution_improvements():
    """Test that tool execution improvements work correctly."""
    print("\n🔧 Testing Tool Execution Improvements...")
    
    agent = MultiRoleAgent()
    session = await agent.create_or_get_session("test_tools")
    
    # Load sample data
    sample_data = create_sample_flight_data()
    session.dataframes = sample_data
    
    # Test Python code executor with incomplete code patterns
    test_incomplete_codes = [
        "average_GPS_quality = 98.7",  # Variable assignment
        "altitude_consistency",  # Single variable
        "analyze GPS data quality",  # Natural language
        "flight_performance = 'high'",  # String assignment
    ]
    
    print("\n📝 Testing Python Code Executor with incomplete/ambiguous inputs:")
    
    for i, code in enumerate(test_incomplete_codes, 1):
        print(f"\nTest {i}: '{code}'")
        try:
            from backend.agent.multi_role_agent import PythonCodeExecutorTool
            from tools.analysis_tools import AnalysisTools
            
            tool = PythonCodeExecutorTool(AnalysisTools(), session)
            result = tool._run(code)
            
            print(f"✅ Result: {result[:200]}...")
            
            # Check if it provided meaningful analysis instead of error
            if "Cannot execute incomplete code" in result:
                print("  ❌ FAIL: Still returning generic error message")
            elif any(keyword in result.lower() for keyword in ['analysis', 'altitude', 'data', 'flight']):
                print("  ✅ PASS: Provided meaningful analysis")
            else:
                print("  ⚠️  PARTIAL: Some response but could be more meaningful")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

async def test_conversation_context_awareness():
    """Test that the agent properly uses conversation context."""
    print("\n💬 Testing Conversation Context Awareness...")
    
    agent = MultiRoleAgent()
    session = await agent.create_or_get_session("test_context")
    
    # Load sample data
    sample_data = create_sample_flight_data()
    session.dataframes = sample_data
    
    # Build conversation history
    conversation_flow = [
        ("What was the maximum altitude?", "altitude query"),
        ("Tell me about the power consumption", "power query"),
        ("How stable was the flight?", "stability query"),
        ("Can you provide more details on that?", "follow-up query"),  # This should reference stability
    ]
    
    print("🗣️  Simulating conversation flow:")
    
    for i, (query, query_type) in enumerate(conversation_flow, 1):
        print(f"\n--- Turn {i}: {query} ({query_type}) ---")
        
        try:
            response = await agent.chat(session.session_id, query)
            print(f"Response: {response[:300]}...")
            
            # For the follow-up query, check if it references previous context
            if query_type == "follow-up query":
                response_lower = response.lower()
                context_keywords = ['stability', 'stable', 'attitude', 'control', 'vibration']
                context_score = sum(1 for keyword in context_keywords if keyword in response_lower)
                
                print(f"\n📊 Context Analysis:")
                print(f"  - Context-relevant keywords: {context_score} found")
                
                if context_score >= 2:
                    print("  ✅ PASS: Successfully used conversation context")
                else:
                    print("  ❌ FAIL: Did not reference previous context appropriately")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def test_planner_improvements():
    """Test that the planner selects more appropriate tools for contextual queries."""
    print("\n🎯 Testing Planner Tool Selection Improvements...")
    
    agent = MultiRoleAgent()
    session = await agent.create_or_get_session("test_planner")
    
    # Load sample data
    sample_data = create_sample_flight_data()
    session.dataframes = sample_data
    
    # Test queries that should trigger specific tool combinations
    test_cases = [
        {
            "query": "why was the flight consistently high?",
            "expected_tools": ["analyze_altitude", "execute_python_code"],
            "description": "Altitude consistency query"
        },
        {
            "query": "show me flight stability analysis",
            "expected_tools": ["execute_python_code"],
            "description": "Flight stability query"
        },
        {
            "query": "analyze the altitude performance in detail",
            "expected_tools": ["analyze_altitude", "execute_python_code"],
            "description": "Detailed altitude analysis"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected tools: {test_case['expected_tools']}")
        
        try:
            # Create planner and test tool selection
            planner = agent._create_planner_agent(session)
            planning_task = agent._create_planning_task(test_case['query'], session)
            planning_task.agent = planner
            
            from crewai import Crew, Process
            planning_crew = Crew(
                agents=[planner],
                tasks=[planning_task],
                process=Process.sequential,
                verbose=False
            )
            
            plan_result = planning_crew.kickoff()
            plan_text = str(plan_result)
            execution_plan = agent._parse_execution_plan(plan_text)
            
            print(f"Selected tools: {execution_plan.tool_sequence}")
            print(f"Reasoning: {execution_plan.reasoning}")
            
            # Check if appropriate tools were selected
            selected_tools = execution_plan.tool_sequence
            expected_tools = test_case['expected_tools']
            
            matches = sum(1 for tool in expected_tools if tool in selected_tools)
            
            if matches >= len(expected_tools) * 0.7:  # At least 70% match
                print("  ✅ PASS: Appropriate tools selected")
            else:
                print("  ⚠️  PARTIAL: Some appropriate tools selected")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

async def main():
    """Run the contextual improvement test."""
    print("🧪 MultiRoleAgent Contextual Improvements Test")
    print("="*60)
    
    try:
        # Test the main issue: contextual altitude understanding
        await test_contextual_altitude_understanding()
        
        # Test 2: Tool execution improvements
        await test_tool_execution_improvements()
        
        # Test 3: Conversation context awareness
        await test_conversation_context_awareness()
        
        # Test 4: Planner improvements
        await test_planner_improvements()
        
        print("\n" + "="*60)
        print("🎉 Test completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 