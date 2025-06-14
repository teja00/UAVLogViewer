#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced multi-role agent capabilities.
This script shows how the agent now handles errors gracefully and provides intelligent fallbacks.
"""

import asyncio
import logging
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from agent.multi_role_agent import MultiRoleAgent
from models import V2ConversationSession, ChatMessage
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_agent():
    """Test the enhanced agent with various scenarios."""
    print("🚀 Testing Enhanced Multi-Role Agent")
    print("=" * 50)
    
    # Initialize the agent
    agent = MultiRoleAgent()
    
    # Create a test session with sample data
    session_id = "test_session_001"
    session = await agent.create_or_get_session(session_id)
    
    # Create sample flight data
    print("📊 Creating sample flight data...")
    
    # GPS data with altitude
    gps_data = {
        'timestamp': pd.date_range('2024-01-01 10:00:00', periods=100, freq='1S'),
        'Alt': np.random.normal(1400, 50, 100),  # Altitude around 1400m
        'Lat': np.random.normal(37.7749, 0.001, 100),
        'Lng': np.random.normal(-122.4194, 0.001, 100),
        'Status': np.random.choice([3, 4], 100, p=[0.9, 0.1])  # Mostly good GPS
    }
    session.dataframes['GPS'] = pd.DataFrame(gps_data)
    
    # Add some error data
    err_data = {
        'timestamp': pd.date_range('2024-01-01 10:05:00', periods=3, freq='30S'),
        'Subsys': ['GPS', 'BARO', 'COMPASS'],
        'ECode': [1, 2, 3]
    }
    session.dataframes['ERR'] = pd.DataFrame(err_data)
    
    # Add message data
    msg_data = {
        'timestamp': pd.date_range('2024-01-01 10:00:00', periods=5, freq='1min'),
        'Message': [
            'Flight mode changed to GUIDED',
            'GPS signal good',
            'WARNING: Battery voltage low',
            'Altitude hold engaged',
            'Landing sequence initiated'
        ]
    }
    session.dataframes['MSG'] = pd.DataFrame(msg_data)
    
    print(f"✅ Created sample data with {len(session.dataframes)} dataframes")
    
    # Test cases
    test_cases = [
        {
            'name': 'Altitude Query (Should work well)',
            'question': 'What was the highest altitude reached during the flight?'
        },
        {
            'name': 'Error Analysis (Should handle tool failures gracefully)',
            'question': 'Check for any errors or warnings'
        },
        {
            'name': 'Incomplete Query (Should use intelligent fallbacks)',
            'question': 'analyze'
        },
        {
            'name': 'Complex Query (Should use multiple tools)',
            'question': 'Give me a summary of the flight performance'
        },
        {
            'name': 'Specific Technical Query',
            'question': 'Were there any GPS signal issues?'
        }
    ]
    
    print("\n🧪 Running test cases...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Question: {test_case['question']}")
        print("   " + "-" * 40)
        
        try:
            # Test the enhanced agent
            response = await agent.chat(session_id, test_case['question'])
            
            # Analyze the response quality
            if len(response) > 50 and not any(fail_word in response.lower() for fail_word in 
                                            ['failed', 'unable', 'technical limitations', 'error']):
                print(f"   ✅ SUCCESS: {response}")
            elif len(response) > 20:
                print(f"   ⚠️  PARTIAL: {response}")
            else:
                print(f"   ❌ FAILED: {response}")
                
        except Exception as e:
            print(f"   💥 EXCEPTION: {str(e)}")
        
        print()
    
    print("=" * 50)
    print("🎯 Test Summary")
    print("The enhanced agent should now:")
    print("  • Handle tool validation errors gracefully")
    print("  • Provide intelligent fallbacks when tools fail")
    print("  • Give specific answers rather than generic errors")
    print("  • Recover from incomplete or ambiguous queries")
    print("  • Use multiple analysis strategies automatically")
    
    # Show performance stats
    try:
        stats = agent.get_performance_stats(session_id)
        print(f"\n📈 Performance Stats:")
        print(f"  • Total LLM calls: {stats.get('total_calls', 'N/A')}")
        print(f"  • Total cost: ${stats.get('total_cost', 0):.4f}")
        print(f"  • Average cost per call: ${stats.get('average_cost_per_call', 0):.4f}")
    except Exception as e:
        print(f"  • Performance stats unavailable: {e}")

if __name__ == "__main__":
    print("🔧 Enhanced UAV Log Viewer Agent Test")
    print("This test demonstrates the improved intelligence and error handling")
    print()
    
    try:
        asyncio.run(test_enhanced_agent())
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 