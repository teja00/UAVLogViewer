"""
Tool definitions for OpenAI function calling.
Contains the schemas for all available analysis tools.
"""

from typing import List, Dict, Any


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get the list of tool definitions for OpenAI function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code to analyze flight data and calculate specific metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute for data analysis.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_anomalies",
                "description": "Detect unusual patterns, errors, or anomalies in the flight data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific areas to check for anomalies (e.g., 'GPS', 'altitude', 'vibration').",
                        }
                    },
                    "required": ["focus_areas"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_metrics",
                "description": "Compare different flight parameters or time periods.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Metrics to compare (e.g., 'GPS_altitude', 'BARO_altitude').",
                        },
                        "comparison_type": {
                            "type": "string",
                            "description": "Type of comparison ('correlation', 'difference', 'trend').",
                        }
                    },
                    "required": ["metrics", "comparison_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_insights",
                "description": "Generate comprehensive insights and summary of the flight.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "What to focus on ('overall', 'performance', 'safety', 'efficiency').",
                        }
                    },
                    "required": ["focus"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_flight_events",
                "description": "Detect specific flight events like GPS loss, mode changes, critical alerts with timestamps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of events to detect: 'gps_loss', 'mode_changes', 'critical_alerts', 'power_issues', 'attitude_problems'",
                        }
                    },
                    "required": ["event_types"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_flight_phase",
                "description": "Analyze specific phases of flight (takeoff, cruise, landing) with detailed metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "string",
                            "description": "Flight phase to analyze: 'takeoff', 'cruise', 'landing', 'all'",
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific metrics to focus on: 'altitude', 'speed', 'power', 'stability'",
                        }
                    },
                    "required": ["phase"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_timeline_analysis",
                "description": "Provide a chronological timeline of key events and issues during the flight.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time_resolution": {
                            "type": "string",
                            "description": "Time resolution for timeline: 'seconds', 'minutes', 'auto'",
                        }
                    },
                    "required": ["time_resolution"],
                },
            },
        }
    ] 