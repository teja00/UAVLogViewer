#!/usr/bin/env python3
"""
Integration Example: Using Existing Telemetry Data with Chatbot Backend

This script demonstrates how to integrate the chatbot backend with
telemetry data that has already been parsed by the frontend JavaScript parsers.
"""

import requests
import json
from typing import Dict, Any

class BackendIntegration:
    """
    Helper class for integrating with the UAV Log Viewer Chatbot Backend
    """
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.session_id = None
    
    def create_session(self) -> str:
        """Create a new chat session"""
        try:
            response = requests.post(f"{self.backend_url}/sessions")
            if response.status_code == 200:
                self.session_id = response.json()["session_id"]
                return self.session_id
            else:
                raise Exception(f"Failed to create session: {response.status_code}")
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def send_telemetry_data(self, telemetry_data: Dict[str, Any]) -> bool:
        """
        Send parsed telemetry data to the backend for analysis.
        
        Args:
            telemetry_data: Dictionary containing 'messages' and 'metadata' 
                           as parsed by the frontend JavaScript parsers
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.session_id:
            self.create_session()
        
        try:
            # Use the telemetry analysis endpoint to store the data
            # This simulates what would happen when data is uploaded
            url = f"{self.backend_url}/telemetry/analyze"
            payload = {
                "query": "data_upload",
                "messages": telemetry_data.get("messages", {}),
                "metadata": telemetry_data.get("metadata", {})
            }
            
            response = requests.post(url, json=payload)
            
            # For actual integration, you would also need to update the session
            # with this telemetry data. This would require a new endpoint or
            # modification of existing endpoints.
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending telemetry data: {e}")
            return False
    
    def chat(self, message: str) -> str:
        """
        Send a message to the chatbot and get response
        
        Args:
            message: User's question about the telemetry data
            
        Returns:
            str: Chatbot's response
        """
        if not self.session_id:
            self.create_session()
        
        try:
            payload = {
                "message": message,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{self.backend_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response received")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error sending message: {e}"

def example_telemetry_data():
    """
    Example telemetry data structure that matches what the JavaScript parsers produce.
    This simulates data from a real flight log.
    """
    return {
        "messages": {
            "GPS": {
                "time_boot_ms": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                "lat": [40.7128, 40.7129, 40.7130, 40.7131, 40.7132, 40.7133, 40.7134, 40.7135, 40.7136, 40.7137],
                "lon": [-74.0060, -74.0061, -74.0062, -74.0063, -74.0064, -74.0065, -74.0066, -74.0067, -74.0068, -74.0069],
                "alt": [10.5, 12.3, 15.8, 18.2, 20.1, 22.5, 25.0, 27.3, 30.1, 32.8]
            },
            "ATT": {
                "time_boot_ms": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                "Roll": [0.1, 0.2, -0.1, 0.3, -0.2, 0.15, -0.25, 0.1, -0.05, 0.0],
                "Pitch": [0.05, -0.1, 0.15, -0.05, 0.08, -0.12, 0.18, -0.08, 0.03, -0.02],
                "Yaw": [1.57, 1.58, 1.56, 1.59, 1.55, 1.60, 1.54, 1.61, 1.52, 1.58]
            },
            "MODE": {
                "time_boot_ms": [500, 2500, 4500, 7500],
                "asText": ["STABILIZE", "LOITER", "AUTO", "RTL"]
            },
            "CURR": {
                "time_boot_ms": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                "Volt": [12.6, 12.5, 12.4, 12.3, 12.2, 12.1, 12.0, 11.9, 11.8, 11.7],
                "Curr": [2.1, 2.3, 2.5, 2.2, 2.0, 2.4, 2.6, 2.1, 1.9, 1.8]
            },
            "STAT": {
                "time_boot_ms": [500, 8500],
                "Armed": [1, 0]  # Armed at 500ms, disarmed at 8500ms
            }
        },
        "metadata": {
            "startTime": 1640995200000,  # Unix timestamp in ms
            "vehicleType": "Quadcopter",
            "logType": "bin",
            "duration": 10000,  # 10 seconds
            "messageCount": 44
        }
    }

def demonstrate_integration():
    """
    Demonstrate how to integrate the backend with existing telemetry data
    """
    print("UAV Log Viewer Chatbot Backend Integration Example")
    print("=" * 60)
    
    # Initialize backend integration
    backend = BackendIntegration()
    
    # Create a session
    print("1. Creating chat session...")
    session_id = backend.create_session()
    if session_id:
        print(f"   ✓ Session created: {session_id}")
    else:
        print("   ✗ Failed to create session")
        return
    
    # Get example telemetry data (this would come from your JavaScript parsers)
    print("\n2. Preparing telemetry data...")
    telemetry_data = example_telemetry_data()
    print(f"   ✓ Data prepared with {len(telemetry_data['messages'])} message types")
    
    # Send telemetry data to backend
    print("\n3. Sending telemetry data to backend...")
    if backend.send_telemetry_data(telemetry_data):
        print("   ✓ Telemetry data sent successfully")
    else:
        print("   ✗ Failed to send telemetry data")
    
    # Example chat interactions
    print("\n4. Testing chat interactions...")
    
    questions = [
        "What was the maximum altitude during this flight?",
        "How long was the total flight time?",
        "What flight modes were used?",
        "What was the battery voltage at the end of the flight?",
        "When did the vehicle get armed and disarmed?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n   Question {i}: {question}")
        response = backend.chat(question)
        print(f"   Answer: {response}")
    
    print("\n" + "=" * 60)
    print("Integration demonstration complete!")
    print("\nTo integrate this with your frontend:")
    print("1. Call backend.create_session() when user opens the app")
    print("2. Call backend.send_telemetry_data() after parsing a log file")
    print("3. Use backend.chat() for user questions")

def javascript_integration_example():
    """
    Show how this would be integrated with JavaScript frontend
    """
    js_code = '''
    // JavaScript Frontend Integration Example
    
    class UAVChatbotAPI {
        constructor(backendUrl = 'http://localhost:8000') {
            this.backendUrl = backendUrl;
            this.sessionId = null;
        }
        
        async createSession() {
            const response = await fetch(`${this.backendUrl}/sessions`, {
                method: 'POST'
            });
            const data = await response.json();
            this.sessionId = data.session_id;
            return this.sessionId;
        }
        
        async sendTelemetryData(messages, metadata) {
            // This would be called after your existing parsers finish
            const payload = {
                query: 'data_upload',
                messages: messages,
                metadata: metadata
            };
            
            const response = await fetch(`${this.backendUrl}/telemetry/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            return response.ok;
        }
        
        async chat(message) {
            const payload = {
                message: message,
                session_id: this.sessionId
            };
            
            const response = await fetch(`${this.backendUrl}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            return data.response;
        }
    }
    
    // Usage in your existing Vue.js app:
    // 1. Initialize the API
    const chatAPI = new UAVChatbotAPI();
    
    // 2. In your file upload component, after parsing:
    worker.onmessage = async (event) => {
        if (event.data.messages && event.data.metadata) {
            // Your existing code...
            this.state.messages = event.data.messages;
            this.state.metadata = event.data.metadata;
            
            // Send to chatbot backend
            await chatAPI.createSession();
            await chatAPI.sendTelemetryData(event.data.messages, event.data.metadata);
        }
    };
    
    // 3. In your chat component:
    async sendMessage(userMessage) {
        const response = await chatAPI.chat(userMessage);
        // Display response in chat UI
        this.addMessageToChat('assistant', response);
    }
    '''
    
    print("\nJavaScript Integration Example:")
    print("=" * 40)
    print(js_code)

if __name__ == "__main__":
    demonstrate_integration()
    javascript_integration_example() 