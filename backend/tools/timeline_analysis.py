"""
Timeline analysis for UAV flight data.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from models import V2ConversationSession

logger = logging.getLogger(__name__)


class TimelineAnalyzer:
    """Analyzes flight timeline with chronological events."""

    def get_timeline(self, session: V2ConversationSession, time_resolution: str) -> str:
        """Get chronological timeline of flight events and key metrics."""
        try:
            # Collect all timestamped events from different data sources
            timeline_events = []
            
            # Get GPS timeline events
            if 'GPS' in session.dataframes:
                gps_events = self._get_gps_timeline_events(session.dataframes['GPS'])
                timeline_events.extend(gps_events)
            
            # Get mode change events
            if 'MODE' in session.dataframes:
                mode_events = self._get_mode_timeline_events(session.dataframes['MODE'])
                timeline_events.extend(mode_events)
            
            # Get power/battery events
            if 'CURR' in session.dataframes:
                power_events = self._get_power_timeline_events(session.dataframes['CURR'])
                timeline_events.extend(power_events)
            
            # Get message/alert events
            if 'MSG' in session.dataframes:
                msg_events = self._get_message_timeline_events(session.dataframes['MSG'])
                timeline_events.extend(msg_events)
            
            # Get error events
            if 'ERR' in session.dataframes:
                err_events = self._get_error_timeline_events(session.dataframes['ERR'])
                timeline_events.extend(err_events)
            
            # Get attitude events
            if 'ATT' in session.dataframes:
                att_events = self._get_attitude_timeline_events(session.dataframes['ATT'])
                timeline_events.extend(att_events)
            
            if not timeline_events:
                return "No timestamped events found in the flight data for timeline analysis."
            
            # Sort events chronologically
            timeline_events.sort(key=lambda x: x.get('timestamp', datetime.min))
            
            # Apply time resolution filtering
            filtered_events = self._filter_by_time_resolution(timeline_events, time_resolution)
            
            # Format timeline response
            return self._format_timeline_response(filtered_events, time_resolution)
            
        except Exception as e:
            logger.error(f"Error generating timeline: {e}")
            return f"Error generating flight timeline: {str(e)}"

    def _get_gps_timeline_events(self, gps_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract GPS-related timeline events."""
        events = []
        
        try:
            if gps_df.empty:
                return events
            
            # GPS signal acquisition/loss events
            if 'Status' in gps_df.columns:
                # Track GPS status changes
                prev_status = None
                for idx, row in gps_df.iterrows():
                    current_status = row['Status']
                    timestamp = row.get('timestamp', None)
                    
                    if prev_status is not None and current_status != prev_status:
                        if current_status >= 3 and prev_status < 3:
                            events.append({
                                'timestamp': timestamp,
                                'type': 'gps_fix_acquired',
                                'description': f'GPS 3D fix acquired (Status: {current_status})',
                                'severity': 'info',
                                'data': {'status': current_status}
                            })
                        elif current_status < 3 and prev_status >= 3:
                            events.append({
                                'timestamp': timestamp,
                                'type': 'gps_fix_lost',
                                'description': f'GPS fix lost (Status: {current_status})',
                                'severity': 'warning',
                                'data': {'status': current_status}
                            })
                    
                    prev_status = current_status
                    
                    # Only process first 20 rows to avoid too many events
                    if idx > 20:
                        break
            
            # Altitude milestones
            if 'Alt' in gps_df.columns:
                altitudes = gps_df['Alt'].dropna()
                if not altitudes.empty:
                    max_alt = altitudes.max()
                    max_alt_idx = altitudes.idxmax()
                    max_alt_time = gps_df.loc[max_alt_idx].get('timestamp', None)
                    
                    events.append({
                        'timestamp': max_alt_time,
                        'type': 'altitude_milestone',
                        'description': f'Maximum altitude reached: {max_alt:.1f}m',
                        'severity': 'info',
                        'data': {'altitude': max_alt}
                    })
            
            # Speed milestones
            if 'Spd' in gps_df.columns:
                speeds = gps_df['Spd'].dropna()
                if not speeds.empty:
                    max_speed = speeds.max()
                    max_speed_idx = speeds.idxmax()
                    max_speed_time = gps_df.loc[max_speed_idx].get('timestamp', None)
                    
                    events.append({
                        'timestamp': max_speed_time,
                        'type': 'speed_milestone',
                        'description': f'Maximum speed reached: {max_speed:.1f} m/s',
                        'severity': 'info',
                        'data': {'speed': max_speed}
                    })
            
        except Exception as e:
            logger.error(f"Error getting GPS timeline events: {e}")
        
        return events

    def _get_mode_timeline_events(self, mode_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract mode change timeline events."""
        events = []
        
        try:
            if mode_df.empty:
                return events
            
            for idx, row in mode_df.iterrows():
                timestamp = row.get('timestamp', None)
                mode_text = row.get('asText', f"Mode {row.get('Mode', 'Unknown')}")
                mode_num = row.get('Mode', 'Unknown')
                
                events.append({
                    'timestamp': timestamp,
                    'type': 'mode_change',
                    'description': f'Flight mode changed to {mode_text}',
                    'severity': 'info',
                    'data': {'mode': mode_text, 'mode_num': mode_num}
                })
        
        except Exception as e:
            logger.error(f"Error getting mode timeline events: {e}")
        
        return events

    def _get_power_timeline_events(self, curr_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract power-related timeline events."""
        events = []
        
        try:
            if curr_df.empty:
                return events
            
            # Voltage thresholds
            if 'Volt' in curr_df.columns:
                voltages = curr_df['Volt'].dropna()
                
                # Find critical voltage events
                critical_voltage = 10.5  # Critical low voltage
                warning_voltage = 11.1   # Warning voltage
                
                critical_events = curr_df[curr_df['Volt'] <= critical_voltage]
                warning_events = curr_df[(curr_df['Volt'] <= warning_voltage) & (curr_df['Volt'] > critical_voltage)]
                
                if not critical_events.empty:
                    first_critical = critical_events.iloc[0]
                    events.append({
                        'timestamp': first_critical.get('timestamp', None),
                        'type': 'critical_voltage',
                        'description': f'Critical battery voltage: {first_critical["Volt"]:.1f}V',
                        'severity': 'critical',
                        'data': {'voltage': first_critical['Volt']}
                    })
                
                if not warning_events.empty:
                    first_warning = warning_events.iloc[0]
                    events.append({
                        'timestamp': first_warning.get('timestamp', None),
                        'type': 'low_voltage_warning',
                        'description': f'Low battery voltage warning: {first_warning["Volt"]:.1f}V',
                        'severity': 'warning',
                        'data': {'voltage': first_warning['Volt']}
                    })
            
            # Current spikes
            if 'Curr' in curr_df.columns:
                currents = curr_df['Curr'].dropna()
                if not currents.empty:
                    avg_current = currents.mean()
                    spike_threshold = avg_current * 2.5  # 2.5x average current
                    
                    current_spikes = curr_df[curr_df['Curr'] >= spike_threshold]
                    if not current_spikes.empty:
                        first_spike = current_spikes.iloc[0]
                        events.append({
                            'timestamp': first_spike.get('timestamp', None),
                            'type': 'current_spike',
                            'description': f'High current spike: {first_spike["Curr"]:.1f}A (avg: {avg_current:.1f}A)',
                            'severity': 'warning',
                            'data': {'current': first_spike['Curr'], 'average': avg_current}
                        })
        
        except Exception as e:
            logger.error(f"Error getting power timeline events: {e}")
        
        return events

    def _get_message_timeline_events(self, msg_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract message/alert timeline events."""
        events = []
        
        try:
            if msg_df.empty or 'Message' not in msg_df.columns:
                return events
            
            # Priority keywords for different severity levels
            critical_keywords = ['ERROR', 'CRITICAL', 'FAIL', 'EMERGENCY']
            warning_keywords = ['WARNING', 'ALERT', 'CAUTION']
            
            for idx, row in msg_df.iterrows():
                message = str(row['Message'])
                timestamp = row.get('timestamp', None)
                
                # Determine severity
                severity = 'info'
                message_upper = message.upper()
                
                for keyword in critical_keywords:
                    if keyword in message_upper:
                        severity = 'critical'
                        break
                
                if severity == 'info':
                    for keyword in warning_keywords:
                        if keyword in message_upper:
                            severity = 'warning'
                            break
                
                events.append({
                    'timestamp': timestamp,
                    'type': 'system_message',
                    'description': f'System message: {message[:60]}...' if len(message) > 60 else f'System message: {message}',
                    'severity': severity,
                    'data': {'full_message': message}
                })
        
        except Exception as e:
            logger.error(f"Error getting message timeline events: {e}")
        
        return events

    def _get_error_timeline_events(self, err_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract error timeline events."""
        events = []
        
        try:
            if err_df.empty:
                return events
            
            for idx, row in err_df.iterrows():
                timestamp = row.get('timestamp', None)
                subsys = row.get('Subsys', 'Unknown')
                ecode = row.get('ECode', 'Unknown')
                
                events.append({
                    'timestamp': timestamp,
                    'type': 'system_error',
                    'description': f'System error: {subsys} subsystem, error code {ecode}',
                    'severity': 'critical',
                    'data': {'subsystem': subsys, 'error_code': ecode}
                })
        
        except Exception as e:
            logger.error(f"Error getting error timeline events: {e}")
        
        return events

    def _get_attitude_timeline_events(self, att_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract attitude-related timeline events."""
        events = []
        
        try:
            if att_df.empty:
                return events
            
            # Check for extreme attitude events
            extreme_threshold = 45  # degrees
            
            for angle in ['Roll', 'Pitch']:
                if angle in att_df.columns:
                    extreme_attitudes = att_df[att_df[angle].abs() > extreme_threshold]
                    
                    if not extreme_attitudes.empty:
                        first_extreme = extreme_attitudes.iloc[0]
                        angle_value = first_extreme[angle]
                        
                        events.append({
                            'timestamp': first_extreme.get('timestamp', None),
                            'type': 'extreme_attitude',
                            'description': f'Extreme {angle.lower()} angle: {angle_value:.1f}°',
                            'severity': 'warning',
                            'data': {'angle_type': angle, 'angle_value': angle_value}
                        })
        
        except Exception as e:
            logger.error(f"Error getting attitude timeline events: {e}")
        
        return events

    def _filter_by_time_resolution(self, events: List[Dict[str, Any]], time_resolution: str) -> List[Dict[str, Any]]:
        """Filter events based on time resolution."""
        try:
            if time_resolution.lower() == 'auto':
                # Auto-determine resolution based on flight duration
                timestamps = [event['timestamp'] for event in events if event.get('timestamp')]
                if timestamps:
                    duration = (max(timestamps) - min(timestamps)).total_seconds()
                    if duration > 3600:  # > 1 hour
                        time_resolution = 'minutes'
                    else:
                        time_resolution = 'seconds'
                else:
                    time_resolution = 'all'
            
            if time_resolution.lower() == 'all':
                return events
            elif time_resolution.lower() == 'minutes':
                # Group events by minute
                return self._group_events_by_minute(events)
            elif time_resolution.lower() == 'seconds':
                return events  # Return all events with second precision
            else:
                return events
        
        except Exception as e:
            logger.error(f"Error filtering by time resolution: {e}")
            return events

    def _group_events_by_minute(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group events by minute intervals."""
        try:
            minute_groups = {}
            
            for event in events:
                timestamp = event.get('timestamp')
                if timestamp:
                    # Round to nearest minute
                    minute_key = timestamp.replace(second=0, microsecond=0)
                    if minute_key not in minute_groups:
                        minute_groups[minute_key] = []
                    minute_groups[minute_key].append(event)
            
            # Create summary events for each minute
            grouped_events = []
            for minute, minute_events in sorted(minute_groups.items()):
                if len(minute_events) == 1:
                    grouped_events.append(minute_events[0])
                else:
                    # Create summary event
                    event_types = [event['type'] for event in minute_events]
                    descriptions = [event['description'] for event in minute_events]
                    
                    summary_event = {
                        'timestamp': minute,
                        'type': 'minute_summary',
                        'description': f'{len(minute_events)} events: {", ".join(event_types[:3])}{"..." if len(event_types) > 3 else ""}',
                        'severity': 'info',
                        'data': {'events': minute_events}
                    }
                    grouped_events.append(summary_event)
            
            return grouped_events
        
        except Exception as e:
            logger.error(f"Error grouping events by minute: {e}")
            return events

    def _format_timeline_response(self, events: List[Dict[str, Any]], time_resolution: str) -> str:
        """Format the timeline response."""
        try:
            if not events:
                return "No events found for timeline analysis."
            
            response_parts = []
            response_parts.append(f"=== FLIGHT TIMELINE ({time_resolution.upper()} RESOLUTION) ===")
            response_parts.append(f"Total events: {len(events)}")
            
            # Get flight duration if possible
            timestamps = [event['timestamp'] for event in events if event.get('timestamp')]
            if timestamps:
                duration = (max(timestamps) - min(timestamps)).total_seconds()
                response_parts.append(f"Flight duration: {duration/60:.1f} minutes")
            
            response_parts.append("\nCHRONOLOGICAL EVENTS:")
            
            # Format each event
            for i, event in enumerate(events[:25], 1):  # Limit to first 25 events
                timestamp = event.get('timestamp')
                if timestamp and isinstance(timestamp, datetime):
                    time_str = timestamp.strftime('%H:%M:%S')
                else:
                    time_str = 'Unknown time'
                
                severity_indicator = {
                    'critical': '🔴',
                    'warning': '🟡',
                    'info': '🔵'
                }.get(event.get('severity', 'info'), '⚪')
                
                response_parts.append(f"{i:2d}. {time_str} {severity_indicator} {event['description']}")
            
            if len(events) > 25:
                response_parts.append(f"\n... and {len(events) - 25} more events")
            
            # Add summary statistics
            event_counts = {}
            severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
            
            for event in events:
                event_type = event.get('type', 'unknown')
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                severity = event.get('severity', 'info')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            response_parts.append("\nEVENT SUMMARY:")
            response_parts.append(f"• Critical events: {severity_counts['critical']}")
            response_parts.append(f"• Warning events: {severity_counts['warning']}")
            response_parts.append(f"• Info events: {severity_counts['info']}")
            
            # Top event types
            top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            response_parts.append("\nTOP EVENT TYPES:")
            for event_type, count in top_events:
                response_parts.append(f"• {event_type}: {count}")
            
            return "\n".join(response_parts)
        
        except Exception as e:
            logger.error(f"Error formatting timeline response: {e}")
            return f"Error formatting timeline: {str(e)}" 