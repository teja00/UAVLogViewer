"""
Event detection for UAV flight data.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from models import V2ConversationSession

logger = logging.getLogger(__name__)


class EventDetector:
    """Detects flight events in UAV data."""

    def detect_events(self, session: V2ConversationSession, event_types: List[str]) -> str:
        """Detect specific flight events with timestamps and details."""
        try:
            all_events = []
            
            for event_type in event_types:
                if event_type.lower() == 'gps_loss':
                    events = self._detect_gps_loss_events(session.dataframes.get('GPS'))
                    all_events.extend(events)
                elif event_type.lower() == 'mode_changes':
                    events = self._detect_mode_changes(session.dataframes.get('MODE'))
                    all_events.extend(events)
                elif event_type.lower() == 'critical_alerts':
                    events = self._detect_critical_alerts(session.dataframes.get('MSG'))
                    all_events.extend(events)
                elif event_type.lower() == 'power_issues':
                    events = self._detect_power_issues(session.dataframes.get('CURR'))
                    all_events.extend(events)
                elif event_type.lower() == 'attitude_problems':
                    events = self._detect_attitude_problems(session.dataframes.get('ATT'))
                    all_events.extend(events)
                else:
                    # Generic event detection for other types
                    events = self._detect_generic_events(session, event_type)
                    all_events.extend(events)
            
            if not all_events:
                return f"No events of type(s) {', '.join(event_types)} detected in the flight data."
            
            # Sort events by timestamp if available, handling None values
            all_events.sort(key=lambda x: x.get('timestamp') if x.get('timestamp') is not None else datetime.min)
            
            # Format response
            response_parts = [f"Detected {len(all_events)} flight events:"]
            
            for i, event in enumerate(all_events, 1):
                timestamp_str = ""
                if 'timestamp' in event and event['timestamp']:
                    if isinstance(event['timestamp'], datetime):
                        timestamp_str = f" at {event['timestamp'].strftime('%H:%M:%S')}"
                    else:
                        timestamp_str = f" at {str(event['timestamp'])}"
                
                response_parts.append(f"{i}. {event['type'].upper()}: {event['description']}{timestamp_str}")
                
                if 'details' in event and event['details']:
                    response_parts.append(f"   Details: {event['details']}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error detecting flight events: {e}")
            return f"Error detecting flight events: {str(e)}"

    def _detect_gps_loss_events(self, gps_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect GPS signal loss events."""
        events = []
        if gps_df is None or gps_df.empty:
            return events
        
        try:
            if 'Status' in gps_df.columns:
                # Find where GPS status drops below 3 (3D fix)
                signal_loss = gps_df[gps_df['Status'] < 3]
                
                if not signal_loss.empty:
                    first_loss = signal_loss.iloc[0]
                    timestamp = first_loss.get('timestamp', None)
                    status = first_loss['Status']
                    
                    events.append({
                        'type': 'gps_loss',
                        'description': f'GPS signal degraded (Status: {status})',
                        'timestamp': timestamp,
                        'details': f'GPS status dropped to {status} (normal is 3+)',
                        'severity': 'critical'
                    })
                    
                    # Count total signal loss instances
                    if len(signal_loss) > 1:
                        events.append({
                            'type': 'gps_loss',
                            'description': f'GPS signal lost {len(signal_loss)} times during flight',
                            'timestamp': None,
                            'details': f'Multiple GPS signal degradation events',
                            'severity': 'warning'
                        })
            
            # Check satellite count issues
            if 'NSats' in gps_df.columns:
                low_sats = gps_df[gps_df['NSats'] < 6]
                if not low_sats.empty:
                    min_sats = gps_df['NSats'].min()
                    events.append({
                        'type': 'gps_sats',
                        'description': f'Low satellite count detected (min: {min_sats})',
                        'timestamp': low_sats.iloc[0].get('timestamp', None),
                        'details': 'Minimum 6 satellites recommended for reliable GPS',
                        'severity': 'warning' if min_sats >= 4 else 'critical'
                    })
        
        except Exception as e:
            logger.error(f"Error detecting GPS events: {e}")
        
        return events

    def _detect_mode_changes(self, mode_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect flight mode changes."""
        events = []
        if mode_df is None or mode_df.empty:
            return events
        
        try:
            mode_changes = len(mode_df)
            if mode_changes > 0:
                events.append({
                    'type': 'mode_changes',
                    'description': f'{mode_changes} flight mode changes detected',
                    'timestamp': mode_df.iloc[0].get('timestamp', None),
                    'details': 'Multiple mode changes may indicate pilot input or autonomous responses',
                    'severity': 'info'
                })
                
                # Detail each mode change if asText is available
                if 'asText' in mode_df.columns:
                    for idx, row in mode_df.head(5).iterrows():  # Show first 5 changes
                        mode_name = row.get('asText', f"Mode {row.get('Mode', 'Unknown')}")
                        timestamp = row.get('timestamp', None)
                        events.append({
                            'type': 'mode_change',
                            'description': f'Changed to {mode_name}',
                            'timestamp': timestamp,
                            'details': f'Flight mode switched to {mode_name}',
                            'severity': 'info'
                        })
        
        except Exception as e:
            logger.error(f"Error detecting mode changes: {e}")
        
        return events

    def _detect_critical_alerts(self, msg_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect critical alerts from MSG data."""
        events = []
        if msg_df is None or msg_df.empty:
            return events
        
        try:
            critical_keywords = ['ERROR', 'CRITICAL', 'FAIL', 'EMERGENCY', 'ALERT', 'WARNING']
            
            if 'Message' in msg_df.columns:
                for idx, row in msg_df.iterrows():
                    message = str(row['Message']).upper()
                    
                    for keyword in critical_keywords:
                        if keyword in message:
                            severity = 'critical' if keyword in ['ERROR', 'CRITICAL', 'FAIL', 'EMERGENCY'] else 'warning'
                            events.append({
                                'type': 'alert',
                                'description': f'{keyword.lower().title()} message: {row["Message"][:50]}...',
                                'timestamp': row.get('timestamp', None),
                                'details': f'System message: {row["Message"]}',
                                'severity': severity
                            })
                            break  # Only one event per message
        
        except Exception as e:
            logger.error(f"Error detecting critical alerts: {e}")
        
        return events

    def _detect_power_issues(self, curr_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect power-related issues."""
        events = []
        if curr_df is None or curr_df.empty:
            return events
        
        try:
            if 'Volt' in curr_df.columns:
                min_voltage = curr_df['Volt'].min()
                if min_voltage < 10.5:
                    events.append({
                        'type': 'power_issue',
                        'description': f'Critical low battery voltage: {min_voltage:.1f}V',
                        'timestamp': curr_df[curr_df['Volt'] == min_voltage].iloc[0].get('timestamp', None),
                        'details': 'Battery voltage below safe operating threshold',
                        'severity': 'critical'
                    })
                elif min_voltage < 11.1:
                    events.append({
                        'type': 'power_issue',
                        'description': f'Low battery voltage warning: {min_voltage:.1f}V',
                        'timestamp': curr_df[curr_df['Volt'] == min_voltage].iloc[0].get('timestamp', None),
                        'details': 'Battery voltage getting low',
                        'severity': 'warning'
                    })
            
            if 'Curr' in curr_df.columns:
                max_current = curr_df['Curr'].max()
                avg_current = curr_df['Curr'].mean()
                if max_current > avg_current * 3:
                    events.append({
                        'type': 'power_issue',
                        'description': f'High current spike detected: {max_current:.1f}A',
                        'timestamp': curr_df[curr_df['Curr'] == max_current].iloc[0].get('timestamp', None),
                        'details': f'Current spike {max_current:.1f}A vs average {avg_current:.1f}A',
                        'severity': 'warning'
                    })
        
        except Exception as e:
            logger.error(f"Error detecting power issues: {e}")
        
        return events

    def _detect_attitude_problems(self, att_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect attitude control problems."""
        events = []
        if att_df is None or att_df.empty:
            return events
        
        try:
            for angle in ['Roll', 'Pitch']:
                if angle in att_df.columns:
                    max_angle = att_df[angle].abs().max()
                    if max_angle > 45:
                        events.append({
                            'type': 'attitude_problem',
                            'description': f'Extreme {angle.lower()} angle: {max_angle:.1f}°',
                            'timestamp': att_df[att_df[angle].abs() == max_angle].iloc[0].get('timestamp', None),
                            'details': f'{angle} exceeded safe limits',
                            'severity': 'critical'
                        })
                    elif max_angle > 30:
                        events.append({
                            'type': 'attitude_problem',
                            'description': f'High {angle.lower()} angle: {max_angle:.1f}°',
                            'timestamp': att_df[att_df[angle].abs() == max_angle].iloc[0].get('timestamp', None),
                            'details': f'{angle} approaching limits',
                            'severity': 'warning'
                        })
        
        except Exception as e:
            logger.error(f"Error detecting attitude problems: {e}")
        
        return events

    def _detect_generic_events(self, session: V2ConversationSession, event_type: str) -> List[Dict[str, Any]]:
        """Generic event detection for unspecified types."""
        events = []
        
        try:
            # Look for event_type in dataframe names or common patterns
            event_type_upper = event_type.upper()
            
            if event_type_upper in session.dataframes:
                df = session.dataframes[event_type_upper]
                events.append({
                    'type': 'data_available',
                    'description': f'{event_type} data found with {len(df)} records',
                    'timestamp': None,
                    'details': f'Dataframe {event_type_upper} contains {len(df)} records',
                    'severity': 'info'
                })
        
        except Exception as e:
            logger.error(f"Error in generic event detection: {e}")
        
        return events 