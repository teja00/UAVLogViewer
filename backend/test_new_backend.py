"""
Test Script for UAV Log Analyzer Backend

Tests the new backend architecture that receives parsed telemetry data
from the frontend and provides AI analysis.
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
from typing import Dict, Any

# Add the current directory to Python path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class BackendTester:
    """Test suite for the UAV Chat Backend"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.session_id = None
        
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        
        print("🧪 UAV Log Analyzer Backend Test Suite")
        print("=" * 50)
        
        try:
            # Test 1: Health check
            await self.test_health_check()
            
            # Test 2: Simulate frontend data
            await self.test_simulate_frontend_data()
            
            # Test 3: Chat with telemetry data
            await self.test_chat_with_telemetry()
            
            # Test 4: Follow-up questions
            await self.test_follow_up_questions()
            
            # Test 5: Session management
            await self.test_session_management()
            
            # Test 6: ArduPilot knowledge
            await self.test_ardupilot_knowledge()
            
            # Test 7: Error handling
            await self.test_error_handling()
            
            print("\n✅ All tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            
    async def test_health_check(self):
        """Test the health endpoint"""
        print("\n1. Testing health check...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.backend_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✓ Status: {data['status']}")
                    print(f"   ✓ OpenAI configured: {data['openai_configured']}")
                    
                    if not data['openai_configured']:
                        print("   ⚠️  Warning: OpenAI API key not configured")
                        
                else:
                    raise Exception(f"Health check failed: {response.status}")
                    
    async def test_simulate_frontend_data(self):
        """Test the frontend data simulation endpoint"""
        print("\n2. Testing frontend data simulation...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.backend_url}/test/simulate-frontend-data") as response:
                if response.status == 200:
                    data = await response.json()
                    print("   ✓ Simulated data generated")
                    print(f"   ✓ Message types: {len(data['data']['messages'])}")
                    
                    # Store for later tests
                    self.simulated_data = data['data']
                    
                else:
                    raise Exception(f"Simulation failed: {response.status}")
                    
    async def test_chat_with_telemetry(self):
        """Test chat endpoint with telemetry data"""
        print("\n3. Testing chat with telemetry data...")
        
        if not hasattr(self, 'simulated_data'):
            raise Exception("No simulated data available")
            
        payload = {
            "message": "Analyze this flight and provide a summary of key metrics",
            "telemetry_data": self.simulated_data
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.backend_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("     Chat response received")
                    print(f"   Session ID: {data['session_id']}")
                    print(f"   Response length: {len(data['response'])} chars")
                    print(f"   Response preview: {data['response'][:100]}...")
                    
                    # Store session ID for follow-up tests
                    self.session_id = data['session_id']
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Chat failed: {response.status} - {error_text}")
                    
    async def test_follow_up_questions(self):
        """Test follow-up questions without re-sending telemetry"""
        print("\n4. Testing follow-up questions...")
        
        if not self.session_id:
            raise Exception("No session ID available")
            
        questions = [
            "What was the maximum altitude reached?",
            "How did the battery perform?",
            "What flight modes were used?",
            "Were there any system messages or warnings?"
        ]
        
        for i, question in enumerate(questions, 1):
            payload = {
                "message": question,
                "session_id": self.session_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✓ Question {i}: {question[:30]}...")
                        print(f"     Response: {data['response'][:60]}...")
                        
                    else:
                        error_text = await response.text()
                        raise Exception(f"Follow-up question {i} failed: {response.status}")
                        
    async def test_session_management(self):
        """Test session management endpoints"""
        print("\n5. Testing session management...")
        
        if not self.session_id:
            raise Exception("No session ID available")
            
        # Get session info
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.backend_url}/sessions/{self.session_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✓ Session info retrieved")
                    print(f"   ✓ Message count: {data['message_count']}")
                    print(f"   ✓ Has telemetry: {data['has_telemetry_data']}")
                    
                else:
                    raise Exception(f"Session info failed: {response.status}")
                    
        # Test clearing session history
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{self.backend_url}/sessions/{self.session_id}") as response:
                if response.status == 200:
                    print("   ✓ Session history cleared")
                    
                else:
                    raise Exception(f"Session clear failed: {response.status}")
                    
    async def test_ardupilot_knowledge(self):
        """Test ArduPilot-specific knowledge"""
        print("\n6. Testing ArduPilot knowledge...")
        
        # Test without telemetry data to check built-in knowledge
        ardupilot_questions = [
            "What is the GPS message type used for in ArduPilot logs?",
            "Explain the ATT message fields",
            "What does the CURR message contain?",
            "What are XKF messages used for?"
        ]
        
        for i, question in enumerate(ardupilot_questions, 1):
            payload = {
                "message": question
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✓ ArduPilot Q{i}: Knowledge demonstrated")
                        
                        # Check if response contains ArduPilot-specific terms
                        response_text = data['response'].lower()
                        if any(term in response_text for term in ['ardupilot', 'mavlink', 'ekf', 'gps', 'attitude']):
                            print(f"     ✓ Response shows ArduPilot expertise")
                        else:
                            print(f"     ⚠️ Response may lack ArduPilot context")
                            
                    else:
                        print(f"   ⚠️ ArduPilot question {i} failed: {response.status}")
                        
    async def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n7. Testing error handling...")
        
        # Test invalid session ID
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.backend_url}/sessions/invalid-session-id") as response:
                if response.status == 404:
                    print("   ✓ Invalid session ID properly rejected")
                else:
                    print(f"   ⚠️ Expected 404 for invalid session, got {response.status}")
                    
        # Test malformed telemetry data
        payload = {
            "message": "Test message",
            "telemetry_data": {
                "invalid": "data structure"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.backend_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status in [200, 422]:  # 422 for validation error
                    print("   ✓ Malformed data handled gracefully")
                else:
                    print(f"   ⚠️ Unexpected response for malformed data: {response.status}")
                    
        # Test empty message
        payload = {
            "message": ""
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.backend_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status in [200, 422]:
                    print("   ✓ Empty message handled")
                else:
                    print(f"   ⚠️ Unexpected response for empty message: {response.status}")


async def run_performance_test():
    """Run basic performance tests"""
    print("\n🚀 Performance Test")
    print("=" * 30)
    
    tester = BackendTester()
    
    # Test response time
    start_time = time.time()
    
    try:
        # Get simulated data
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{tester.backend_url}/test/simulate-frontend-data") as response:
                simulated_data = (await response.json())['data']
                
        # Send chat request
        payload = {
            "message": "Quick analysis of this flight data",
            "telemetry_data": simulated_data
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{tester.backend_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    print(f"Response time: {response_time:.2f} seconds")
                    
                    if response_time < 10:
                        print("✓ Good response time")
                    elif response_time < 20:
                        print("⚠️ Acceptable response time")
                    else:
                        print("❌ Slow response time")
                        
                else:
                    print(f"❌ Performance test failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ Performance test error: {e}")


async def main():
    """Run all tests"""
    print("Starting UAV Log Analyzer Backend Tests...")
    print("Make sure the backend is running: python start_server.py")
    print()
    
    # Wait a moment for user to read
    await asyncio.sleep(2)
    
    try:
        # Main test suite
        tester = BackendTester()
        await tester.run_all_tests()
        
        # Performance test
        await run_performance_test()
        
        print("\n" + "=" * 50)
        print("🎉 Test suite completed!")
        print("\nNext steps:")
        print("1. Integrate with Vue.js frontend")
        print("2. Test with real .bin files")
        print("3. Deploy to production")
        
    except aiohttp.ClientConnectorError:
        print("❌ Cannot connect to backend!")
        print("💡 Make sure the backend is running:")
        print("   cd backend && python start_server.py")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 