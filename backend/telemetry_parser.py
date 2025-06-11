"""
Telemetry Parser Module

This module provides Python equivalents of the JavaScript parsing logic
found in the frontend. It's designed to work with the same data structures
and provide similar functionality for MAVLink telemetry analysis.
"""

import struct
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

class TelemetryParser:
    """
    Base class for telemetry parsing functionality.
    Provides common methods for data extraction and analysis.
    """
    
    def __init__(self):
        self.messages = {}
        self.metadata = {}
        self.start_time = None
        
    def extract_attitude(self, messages: Dict, source: str = "ATT") -> Dict[int, List[float]]:
        """Extract attitude data (roll, pitch, yaw) from messages."""
        attitudes = {}
        if source in messages:
            attitude_msgs = messages[source]
            if 'time_boot_ms' in attitude_msgs:
                for i, timestamp in enumerate(attitude_msgs['time_boot_ms']):
                    attitudes[int(timestamp)] = [
                        attitude_msgs.get('Roll', [0])[i] if i < len(attitude_msgs.get('Roll', [])) else 0,
                        attitude_msgs.get('Pitch', [0])[i] if i < len(attitude_msgs.get('Pitch', [])) else 0,
                        attitude_msgs.get('Yaw', [0])[i] if i < len(attitude_msgs.get('Yaw', [])) else 0
                    ]
        return attitudes
    
    def extract_flight_modes(self, messages: Dict) -> List[List]:
        """Extract flight mode changes from messages."""
        modes = []
        if 'MODE' in messages:
            msgs = messages['MODE']
            if 'time_boot_ms' in msgs and 'asText' in msgs:
                if msgs['time_boot_ms'] and msgs['asText']:
                    modes = [[msgs['time_boot_ms'][0], msgs['asText'][0]]]
                    for i in range(1, len(msgs['time_boot_ms'])):
                        if (i < len(msgs['asText']) and 
                            msgs['asText'][i] != modes[-1][1] and 
                            msgs['asText'][i] is not None):
                            modes.append([msgs['time_boot_ms'][i], msgs['asText'][i]])
        return modes
    
    def extract_gps_data(self, messages: Dict) -> Dict[str, Any]:
        """Extract GPS trajectory and position data."""
        gps_data = {
            'trajectory': [],
            'positions': {},
            'altitude_data': []
        }
        
        # Check for different GPS message types
        gps_sources = ['GLOBAL_POSITION_INT', 'GPS_RAW_INT', 'GPS']
        
        for source in gps_sources:
            if source in messages:
                msgs = messages[source]
                if 'time_boot_ms' in msgs:
                    for i, timestamp in enumerate(msgs['time_boot_ms']):
                        lat = msgs.get('lat', [0])[i] if i < len(msgs.get('lat', [])) else 0
                        lon = msgs.get('lon', [0])[i] if i < len(msgs.get('lon', [])) else 0
                        alt = msgs.get('alt', [0])[i] if i < len(msgs.get('alt', [])) else 0
                        
                        # Convert coordinates if needed (MAVLink uses 1e-7 scaling)
                        if source == 'GLOBAL_POSITION_INT':
                            lat = lat / 1e7 if abs(lat) > 180 else lat
                            lon = lon / 1e7 if abs(lon) > 180 else lon
                        elif source == 'GPS_RAW_INT':
                            lat = lat / 1e7 if abs(lat) > 180 else lat
                            lon = lon / 1e7 if abs(lon) > 180 else lon
                            alt = alt / 1000  # mm to m
                        
                        if lat != 0 and lon != 0:
                            gps_data['trajectory'].append([lon, lat, alt, timestamp])
                            gps_data['positions'][timestamp] = [lon, lat, alt]
                            gps_data['altitude_data'].append([timestamp, alt])
                break  # Use first available source
        
        return gps_data
    
    def extract_battery_data(self, messages: Dict) -> Dict[str, Any]:
        """Extract battery voltage and current data."""
        battery_data = {
            'voltage': [],
            'current': [],
            'remaining': []
        }
        
        battery_sources = ['BATTERY_STATUS', 'SYS_STATUS', 'CURR']
        
        for source in battery_sources:
            if source in messages:
                msgs = messages[source]
                if 'time_boot_ms' in msgs:
                    for i, timestamp in enumerate(msgs['time_boot_ms']):
                        voltage = msgs.get('voltage_battery', msgs.get('Volt', [0]))[i] if i < len(msgs.get('voltage_battery', msgs.get('Volt', []))) else 0
                        current = msgs.get('current_battery', msgs.get('Curr', [0]))[i] if i < len(msgs.get('current_battery', msgs.get('Curr', []))) else 0
                        remaining = msgs.get('battery_remaining', [100])[i] if i < len(msgs.get('battery_remaining', [])) else 100
                        
                        battery_data['voltage'].append([timestamp, voltage])
                        battery_data['current'].append([timestamp, current])
                        battery_data['remaining'].append([timestamp, remaining])
                break
        
        return battery_data
    
    def extract_events(self, messages: Dict) -> List[List]:
        """Extract significant events from the flight log."""
        events = []
        
        # Armed/Disarmed events
        if 'STAT' in messages:
            msgs = messages['STAT']
            if 'time_boot_ms' in msgs and 'Armed' in msgs:
                if msgs['time_boot_ms'] and msgs['Armed']:
                    armed_state = [[msgs['time_boot_ms'][0], 'Armed' if msgs['Armed'][0] == 1 else 'Disarmed']]
                    for i in range(1, len(msgs['time_boot_ms'])):
                        if i < len(msgs['Armed']):
                            new_state = 'Armed' if msgs['Armed'][i] == 1 else 'Disarmed'
                            if new_state != armed_state[-1][1]:
                                armed_state.append([msgs['time_boot_ms'][i], new_state])
                    events.extend(armed_state)
        
        # Event messages
        if 'EV' in messages:
            msgs = messages['EV']
            if 'time_boot_ms' in msgs and 'Id' in msgs:
                event_names = {
                    10: 'ARMED', 11: 'DISARMED', 25: 'SET_HOME',
                    18: 'LAND_COMPLETE', 19: 'LOST_GPS'
                }
                for i, timestamp in enumerate(msgs['time_boot_ms']):
                    event_id = msgs['Id'][i] if i < len(msgs['Id']) else 0
                    event_name = event_names.get(event_id, f'EVENT_{event_id}')
                    events.append([timestamp, event_name])
        
        return sorted(events, key=lambda x: x[0])  # Sort by timestamp
    
    def calculate_flight_statistics(self, messages: Dict) -> Dict[str, Any]:
        """Calculate various flight statistics from the telemetry data."""
        stats = {
            'total_flight_time': 0,
            'max_altitude': 0,
            'max_ground_speed': 0,
            'max_air_speed': 0,
            'distance_traveled': 0,
            'battery_consumed': 0,
            'max_battery_temp': 0
        }
        
        # Calculate flight time
        if messages:
            all_timestamps = []
            for msg_type, msg_data in messages.items():
                if 'time_boot_ms' in msg_data and msg_data['time_boot_ms']:
                    all_timestamps.extend(msg_data['time_boot_ms'])
            
            if all_timestamps:
                stats['total_flight_time'] = (max(all_timestamps) - min(all_timestamps)) / 1000.0  # Convert to seconds
        
        # Calculate altitude statistics
        gps_data = self.extract_gps_data(messages)
        if gps_data['altitude_data']:
            altitudes = [alt for _, alt in gps_data['altitude_data']]
            stats['max_altitude'] = max(altitudes) if altitudes else 0
        
        # Calculate speed statistics
        if 'VFR_HUD' in messages:
            msgs = messages['VFR_HUD']
            if 'groundspeed' in msgs:
                stats['max_ground_speed'] = max(msgs['groundspeed']) if msgs['groundspeed'] else 0
            if 'airspeed' in msgs:
                stats['max_air_speed'] = max(msgs['airspeed']) if msgs['airspeed'] else 0
        
        # Calculate battery statistics
        battery_data = self.extract_battery_data(messages)
        if battery_data['voltage']:
            voltages = [v for _, v in battery_data['voltage']]
            stats['max_battery_temp'] = max(voltages) if voltages else 0
        
        return stats
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the parsed telemetry data."""
        if not self.messages:
            return {"error": "No telemetry data available"}
        
        summary = {
            'message_types': list(self.messages.keys()),
            'total_messages': sum(len(msg.get('time_boot_ms', [])) for msg in self.messages.values()),
            'flight_statistics': self.calculate_flight_statistics(self.messages),
            'attitude_data': self.extract_attitude(self.messages),
            'flight_modes': self.extract_flight_modes(self.messages),
            'gps_data': self.extract_gps_data(self.messages),
            'battery_data': self.extract_battery_data(self.messages),
            'events': self.extract_events(self.messages),
            'metadata': self.metadata
        }
        
        return summary
    
    def query_telemetry_data(self, query: str) -> Dict[str, Any]:
        """
        Query telemetry data based on natural language questions.
        This method provides structured data that can be used by the LLM.
        """
        query_lower = query.lower()
        result = {"query": query, "data": {}}
        
        # Altitude queries
        if any(word in query_lower for word in ['altitude', 'height', 'high']):
            gps_data = self.extract_gps_data(self.messages)
            if gps_data['altitude_data']:
                altitudes = [alt for _, alt in gps_data['altitude_data']]
                result["data"]["altitude"] = {
                    "max_altitude": max(altitudes) if altitudes else 0,
                    "min_altitude": min(altitudes) if altitudes else 0,
                    "altitude_data": gps_data['altitude_data'][:10]  # First 10 points
                }
        
        # Battery queries
        if any(word in query_lower for word in ['battery', 'voltage', 'current', 'power']):
            battery_data = self.extract_battery_data(self.messages)
            result["data"]["battery"] = {
                "voltage_data": battery_data['voltage'][:10],
                "current_data": battery_data['current'][:10],
                "remaining_data": battery_data['remaining'][:10]
            }
        
        # Flight time queries
        if any(word in query_lower for word in ['time', 'duration', 'long']):
            stats = self.calculate_flight_statistics(self.messages)
            result["data"]["flight_time"] = {
                "total_seconds": stats['total_flight_time'],
                "formatted_time": f"{int(stats['total_flight_time'] // 60)}m {int(stats['total_flight_time'] % 60)}s"
            }
        
        # GPS/location queries
        if any(word in query_lower for word in ['gps', 'location', 'position', 'coordinate']):
            gps_data = self.extract_gps_data(self.messages)
            result["data"]["gps"] = {
                "total_positions": len(gps_data['trajectory']),
                "first_position": gps_data['trajectory'][0] if gps_data['trajectory'] else None,
                "last_position": gps_data['trajectory'][-1] if gps_data['trajectory'] else None
            }
        
        # Event queries
        if any(word in query_lower for word in ['event', 'error', 'arm', 'disarm', 'land']):
            events = self.extract_events(self.messages)
            result["data"]["events"] = events
        
        # Flight mode queries
        if any(word in query_lower for word in ['mode', 'flight mode']):
            modes = self.extract_flight_modes(self.messages)
            result["data"]["flight_modes"] = modes
        
        # If no specific data was queried, return summary
        if not result["data"]:
            result["data"]["summary"] = self.get_data_summary()
        
        return result 