#!/usr/bin/env python3
"""
Simple test script for UAV Log Viewer Chatbot Backend

This script tests the basic functionality of the backend API.
"""

import requests
import json
import time
import os
from io import BytesIO

# Backend URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_create_session():
    """Test creating a new session."""
    print("\nTesting session creation...")
    try:
        response = requests.post(f"{BASE_URL}/sessions")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")
        if response.status_code == 200:
            return data.get("session_id")
        return None
    except Exception as e:
        print(f"Session creation failed: {e}")
        return None

def test_file_upload(session_id):
    """Test file upload with simulated data."""
    print("\nTesting file upload...")
    try:
        # Create a dummy file for testing
        dummy_file_content = b"Test binary data for UAV log file simulation"
        files = {"file": ("test_log.bin", BytesIO(dummy_file_content), "application/octet-stream")}
        data = {"session_id": session_id} if session_id else {}
        
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"File upload failed: {e}")
        return False

def test_chat(session_id):
    """Test chat functionality."""
    print("\nTesting chat...")
    
    # Skip chat test if no OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping chat test - no OpenAI API key configured")
        return True
    
    try:
        test_message = "What is the maximum altitude in this flight log?"
        payload = {
            "message": test_message,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"User: {test_message}")
        print(f"Bot: {result.get('response', 'No response')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Chat test failed: {e}")
        return False

def test_session_info(session_id):
    """Test getting session information."""
    print("\nTesting session info...")
    try:
        response = requests.get(f"{BASE_URL}/sessions/{session_id}")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Session info test failed: {e}")
        return False

def test_telemetry_analysis():
    """Test direct telemetry analysis."""
    print("\nTesting telemetry analysis...")
    try:
        sample_telemetry = {
            "query": "altitude analysis",
            "messages": {
                "GPS": {
                    "time_boot_ms": [1000, 2000, 3000],
                    "lat": [40.7128, 40.7129, 40.7130],
                    "lon": [-74.0060, -74.0061, -74.0062],
                    "alt": [10.5, 15.8, 20.1]
                }
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/telemetry/analyze",
            json=sample_telemetry,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Analysis: {json.dumps(result, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Telemetry analysis test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("UAV Log Viewer Chatbot Backend Test Suite")
    print("=" * 50)
    
    # Check if server is running
    print("Checking if backend server is running...")
    time.sleep(1)
    
    if not test_health_check():
        print("\nERROR: Backend server is not running or not responding.")
        print("Please start the server with: python start_server.py")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Create session
    total_tests += 1
    session_id = test_create_session()
    if session_id:
        tests_passed += 1
        print("✓ Session creation test passed")
    else:
        print("✗ Session creation test failed")
    
    # Test 2: File upload
    if session_id:
        total_tests += 1
        if test_file_upload(session_id):
            tests_passed += 1
            print("✓ File upload test passed")
        else:
            print("✗ File upload test failed")
    
    # Test 3: Chat functionality
    if session_id:
        total_tests += 1
        if test_chat(session_id):
            tests_passed += 1
            print("✓ Chat test passed")
        else:
            print("✗ Chat test failed")
    
    # Test 4: Session info
    if session_id:
        total_tests += 1
        if test_session_info(session_id):
            tests_passed += 1
            print("✓ Session info test passed")
        else:
            print("✗ Session info test failed")
    
    # Test 5: Telemetry analysis
    total_tests += 1
    if test_telemetry_analysis():
        tests_passed += 1
        print("✓ Telemetry analysis test passed")
    else:
        print("✗ Telemetry analysis test failed")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\nNext steps:")
    print("1. Configure your OpenAI API key in .env file for full chat functionality")
    print("2. Integrate the backend with your frontend application")
    print("3. Test with real flight log files")

if __name__ == "__main__":
    main() 