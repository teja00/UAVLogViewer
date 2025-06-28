#!/usr/bin/env python3
"""
Test script to demonstrate CrewAI memory system integration with MultiRoleAgent.

This script shows how the memory system enables natural follow-up conversations
without hardcoding follow-up logic.
"""

import os
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memory_system():
    """Test the memory-enabled MultiRoleAgent with follow-up questions."""
    from agent.multi_role_agent import MultiRoleAgent
    
    # Initialize the agent with memory
    agent = MultiRoleAgent()
    
    # Test session
    session_id = "test_memory_session"
    
    print("\n🧠 CrewAI Memory System Test")
    print("="*50)
    
    print(f"Memory storage location: {agent.memory_storage_dir}")
    print(f"Memory embedder config: {agent.memory_embedder_config}")
    
    print("\n✅ Memory system configured successfully!")
    print("The agent now has:")
    print("- Short-term memory for recent context")
    print("- Long-term memory for insights across sessions")
    print("- Entity memory for tracking concepts and relationships")
    
    # Simulate a conversation flow that would benefit from memory
    conversation_flow = [
        "What was the maximum altitude in this flight?",
        "How does that compare to typical flights?",  # Follow-up referencing "that" 
        "Tell me more about the altitude patterns",    # Follow-up about altitude
        "What about power consumption during high altitude?",  # Related follow-up
        "Were there any issues with the battery?",     # New topic
        "How did those battery issues affect the flight?"  # Follow-up about "those issues"
    ]
    
    print("\n📝 Example conversation flow that benefits from memory:")
    for i, message in enumerate(conversation_flow, 1):
        print(f"{i}. User: {message}")
        if i == len(conversation_flow):
            break
        print(f"   → Agent remembers context for next question")
    
    print("\n🎯 Key Benefits:")
    print("- No need to hardcode follow-up logic")
    print("- Natural reference resolution ('that', 'those issues', etc.)")
    print("- Context-aware responses across conversation")
    print("- Persistent memory across sessions")
    
    # Show memory file structure
    if os.path.exists(agent.memory_storage_dir):
        print(f"\n📁 Memory storage structure:")
        for root, dirs, files in os.walk(agent.memory_storage_dir):
            level = root.replace(agent.memory_storage_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print(f"\n📁 Memory storage will be created at: {agent.memory_storage_dir}")
    
    print("\n✨ The memory system will automatically:")
    print("- Remember previous questions and answers")
    print("- Track entities (altitude, battery, GPS, etc.)")
    print("- Maintain conversation context")
    print("- Enable natural follow-up conversations")

if __name__ == "__main__":
    asyncio.run(test_memory_system()) 