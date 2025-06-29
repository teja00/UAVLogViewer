"""
Configuration management for UAV Log Analyzer Backend
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Server Configuration  
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # CORS settings
    cors_origins: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080,http://localhost:8001"
    ).split(",")
    
    # Performance Configuration
    use_optimized_parser: bool = os.getenv("USE_OPTIMIZED_PARSER", "true").lower() == "true"
    parser_max_workers: int = int(os.getenv("PARSER_MAX_WORKERS", "0"))  # 0 means auto-detect
    
    # Intent Classification Configuration
    intents: Dict[str, Dict[str, Any]] = {
        "altitude_analysis": {
            "name": "Altitude Analysis",
            "description": "Questions about altitude, height, maximum/minimum altitudes, climb rates, altitude changes",
            "tools": ["analyze_altitude", "execute_python_code"],
            "confidence_threshold": 0.8,
            "examples": [
                "What was the maximum altitude?",
                "How high did the aircraft fly?",
                "Show me altitude changes during flight",
                "What was the lowest altitude reached?"
            ]
        },
        "power_system_analysis": {
            "name": "Power System Analysis", 
            "description": "Questions about battery, voltage, current, power consumption, temperature",
            "tools": ["execute_python_code"],
            "confidence_threshold": 0.8,
            "examples": [
                "Analyze power consumption",
                "What was the battery temperature?",
                "Show me voltage levels",
                "How much current was consumed?"
            ]
        },
        "flight_events_analysis": {
            "name": "Flight Events Analysis",
            "description": "Questions about errors, warnings, critical events, system failures, alerts",
            "tools": ["detect_flight_events", "execute_python_code"],
            "confidence_threshold": 0.8,
            "examples": [
                "Were there any critical errors?",
                "Show me flight warnings",
                "What errors occurred mid-flight?",
                "Check for system failures"
            ]
        },
        "navigation_analysis": {
            "name": "Navigation Analysis",
            "description": "Questions about GPS, signal quality, position accuracy, navigation health, satellites",
            "tools": ["execute_python_code", "detect_flight_events"],
            "confidence_threshold": 0.8,
            "examples": [
                "How was the GPS signal quality?",
                "Any GPS signal loss?",
                "Check navigation health",
                "How many satellites were visible?"
            ]
        },
        "flight_performance_analysis": {
            "name": "Flight Performance Analysis",
            "description": "Comprehensive flight summaries, overviews, performance metrics, flight duration",
            "tools": ["enhanced_telemetry_analysis", "execute_python_code", "detect_flight_events"],
            "confidence_threshold": 0.7,
            "examples": [
                "Give me a flight summary",
                "Analyze overall flight performance", 
                "How long was the flight?",
                "Provide a comprehensive overview"
            ]
        },
        "anomaly_detection": {
            "name": "Anomaly Detection",
            "description": "Questions about unusual patterns, anomalies, unexpected behavior, data inconsistencies",
            "tools": ["find_anomalies", "enhanced_telemetry_analysis"],
            "confidence_threshold": 0.7,
            "examples": [
                "Find any anomalies in the data",
                "Detect unusual patterns",
                "Check for unexpected behavior",
                "Are there any data inconsistencies?"
            ]
        },
        "flight_phase_analysis": {
            "name": "Flight Phase Analysis", 
            "description": "Questions about specific flight phases like takeoff, landing, cruise, flight modes",
            "tools": ["analyze_flight_phase", "execute_python_code"],
            "confidence_threshold": 0.8,
            "examples": [
                "Analyze the takeoff phase",
                "How was the landing?",
                "Show me cruise performance",
                "What flight modes were used?"
            ]
        },
        "general_code_execution": {
            "name": "General Code Execution",
            "description": "Custom calculations, specific data extraction, complex queries requiring flexible analysis",
            "tools": ["execute_python_code"],
            "confidence_threshold": 0.5,
            "examples": [
                "Calculate the average speed",
                "Extract specific telemetry values",
                "Perform custom analysis",
                "Show me raw data from specific sensors"
            ]
        }
    }
    
    # Intent Classification Settings
    intent_classification_model: str = os.getenv("INTENT_CLASSIFICATION_MODEL", "gpt-3.5-turbo")
    intent_classification_temperature: float = float(os.getenv("INTENT_CLASSIFICATION_TEMPERATURE", "0.1"))
    intent_classification_max_tokens: int = int(os.getenv("INTENT_CLASSIFICATION_MAX_TOKENS", "100"))

def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings() 