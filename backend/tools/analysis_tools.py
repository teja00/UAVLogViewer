"""
Analysis tools for UAV flight data.
Contains implementations of all analysis tools used by the AI agent.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from models import V2ConversationSession

logger = logging.getLogger(__name__)


class AnalysisTools:
    """Collection of analysis tools for UAV flight data."""

    def execute_python_code(self, session: V2ConversationSession, code: str) -> str:
        """Execute Python code with enhanced error handling and user-friendly output."""
        try:
            logger.info(f"Executing AI-generated code for session {session.session_id}:\n{code}")
            
            # Enhanced local scope with more helper functions
            local_scope = {
                "dfs": session.dataframes, 
                "pd": pd, 
                "np": np,
                "session": session
            }
            
            # Add helpful shortcuts for common operations
            local_scope.update({
                "get_max": lambda df, col: df[col].max() if col in df.columns else None,
                "get_min": lambda df, col: df[col].min() if col in df.columns else None,
                "get_stats": lambda df, col: {
                    "max": df[col].max(),
                    "min": df[col].min(), 
                    "mean": df[col].mean(),
                    "std": df[col].std()
                } if col in df.columns else None
            })
            
            code_lines = code.strip().split('\n')
            
            # Execute all lines except the last
            if len(code_lines) > 1:
                exec('\n'.join(code_lines[:-1]), {}, local_scope)
            
            # Evaluate the last line and return the result
            result = eval(code_lines[-1], {}, local_scope)
            
            # Format result based on type and content
            if isinstance(result, (int, float)):
                return f"Result: {result:.2f}"
            elif isinstance(result, dict):
                formatted_result = "\n".join([f"- {k}: {v:.2f}" if isinstance(v, (int, float)) else f"- {k}: {v}" for k, v in result.items()])
                return f"Analysis Results:\n{formatted_result}"
            elif isinstance(result, str):
                # Check if the string already looks like a complete answer
                if any(phrase in result.lower() for phrase in ['the maximum', 'the minimum', 'the total', 'the first', 'no ', 'found', 'detected', 'occurred']):
                    # Return well-formatted strings as-is
                    return result
                else:
                    # Add minimal formatting for simple strings
                    return f"Result: {result}"
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Error executing code: {e}")
            
            # Provide helpful error messages for common issues
            error_str = str(e)
            if "KeyError" in error_str:
                return f"Analysis error: Column or dataframe not found. Available dataframes: {list(session.dataframes.keys())}. Error: {error_str}"
            elif "NameError" in error_str:
                return f"Analysis error: Variable not defined. Make sure to reference dataframes as dfs['DATAFRAME_NAME']. Error: {error_str}"
            else:
                return f"Analysis error: {error_str}"

    def find_anomalies(self, session: V2ConversationSession, focus_areas: List[str]) -> str:
        """Enhanced anomaly detection with temporal analysis and severity assessment."""
        from .anomaly_detection import AnomalyDetector
        detector = AnomalyDetector()
        return detector.detect_anomalies(session, focus_areas)

    def compare_metrics(self, session: V2ConversationSession, metrics: List[str], comparison_type: str) -> str:
        """Compare different metrics in the flight data."""
        try:
            results = []
            
            for metric in metrics:
                if '.' in metric:
                    msg_type, col = metric.split('.', 1)
                    if msg_type in session.dataframes and col in session.dataframes[msg_type].columns:
                        values = session.dataframes[msg_type][col].dropna()
                        results.append({
                            'metric': metric,
                            'mean': values.mean(),
                            'max': values.max(),
                            'min': values.min(),
                            'count': len(values)
                        })
            
            if len(results) < 2:
                return "Need at least 2 valid metrics to compare."
            
            comparison_result = f"Comparison of {len(results)} metrics:\n"
            for result in results:
                comparison_result += f"- {result['metric']}: avg={result['mean']:.2f}, range={result['min']:.2f} to {result['max']:.2f}\n"
            
            return comparison_result
            
        except Exception as e:
            return f"Error comparing metrics: {str(e)}"

    def generate_insights(self, session: V2ConversationSession, focus: str) -> str:
        """Generate comprehensive insights about the flight."""
        from .insights_generator import InsightsGenerator
        generator = InsightsGenerator()
        return generator.generate_insights(session, focus)

    def detect_flight_events(self, session: V2ConversationSession, event_types: List[str]) -> str:
        """Detect specific flight events like GPS loss, mode changes, critical alerts with timestamps."""
        from .event_detection import EventDetector
        detector = EventDetector()
        return detector.detect_events(session, event_types)

    def analyze_flight_phase(self, session: V2ConversationSession, phase: str, metrics: List[str]) -> str:
        """Analyze specific phases of flight (takeoff, cruise, landing) with detailed metrics."""
        from .phase_analysis import PhaseAnalyzer
        analyzer = PhaseAnalyzer()
        return analyzer.analyze_phase(session, phase, metrics)

    def get_timeline_analysis(self, session: V2ConversationSession, time_resolution: str) -> str:
        """Provide a chronological timeline of key events and issues during the flight."""
        from .timeline_analysis import TimelineAnalyzer
        analyzer = TimelineAnalyzer()
        return analyzer.get_timeline(session, time_resolution)

    # Private helper methods for anomaly analysis
    def _analyze_gps_anomalies(self, gps_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detailed GPS-specific anomaly analysis."""
        issues = {'critical': [], 'warning': [], 'info': []}
        
        try:
            # Check for GPS signal loss
            if 'Status' in gps_df.columns:
                signal_loss_points = gps_df[gps_df['Status'] < 3]  # Less than 3D fix
                if len(signal_loss_points) > 0:
                    first_loss = signal_loss_points.iloc[0]
                    if 'timestamp' in gps_df.columns:
                        loss_time = first_loss['timestamp'].strftime("%H:%M:%S")
                        issues['critical'].append(f"GPS signal lost at {loss_time} (Status: {first_loss['Status']})")
                    else:
                        issues['critical'].append(f"GPS signal degraded {len(signal_loss_points)} times")
            
            # Check satellite count
            if 'NSats' in gps_df.columns:
                low_sat_points = gps_df[gps_df['NSats'] < 6]
                if len(low_sat_points) > 0:
                    min_sats = gps_df['NSats'].min()
                    issues['warning'].append(f"Low satellite count: minimum {min_sats} satellites")
            
            # Check HDOP (horizontal dilution of precision)
            if 'HDop' in gps_df.columns:
                high_hdop = gps_df[gps_df['HDop'] > 2.0]
                if len(high_hdop) > 0:
                    max_hdop = gps_df['HDop'].max()
                    issues['warning'].append(f"Poor GPS precision: max HDOP {max_hdop:.1f}")
            
            # Check for altitude jumps
            if 'Alt' in gps_df.columns and len(gps_df) > 1:
                alt_diff = gps_df['Alt'].diff().abs()
                large_jumps = alt_diff[alt_diff > 50]  # 50m jumps
                if len(large_jumps) > 0:
                    max_jump = alt_diff.max()
                    issues['warning'].append(f"GPS altitude jumps detected: max {max_jump:.1f}m")
            
            # Check ground speed anomalies
            if 'Spd' in gps_df.columns:
                high_speed = gps_df[gps_df['Spd'] > 50]  # > 50 m/s seems excessive
                if len(high_speed) > 0:
                    max_speed = gps_df['Spd'].max()
                    issues['warning'].append(f"High speed readings: max {max_speed:.1f} m/s")
            
        except Exception as e:
            issues['warning'].append(f"GPS analysis error: {str(e)}")
        
        return issues

    def _analyze_attitude_anomalies(self, att_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze attitude control anomalies."""
        issues = {'critical': [], 'warning': [], 'info': []}
        
        try:
            # Check for excessive roll/pitch
            for angle in ['Roll', 'Pitch']:
                if angle in att_df.columns:
                    max_angle = att_df[angle].abs().max()
                    if max_angle > 45:  # Degrees
                        issues['critical'].append(f"Extreme {angle.lower()}: {max_angle:.1f}°")
                    elif max_angle > 30:
                        issues['warning'].append(f"High {angle.lower()}: {max_angle:.1f}°")
            
            # Check attitude error
            if 'ErrRP' in att_df.columns:
                high_error = att_df[att_df['ErrRP'].abs() > 20]
                if len(high_error) > 0:
                    max_error = att_df['ErrRP'].abs().max()
                    issues['warning'].append(f"Attitude control errors: max {max_error:.1f}°")
            
            # Check for oscillations
            if 'Roll' in att_df.columns and len(att_df) > 10:
                roll_std = att_df['Roll'].rolling(window=10).std().max()
                if roll_std > 10:
                    issues['warning'].append(f"Roll oscillations detected: std {roll_std:.1f}°")
                    
        except Exception as e:
            issues['warning'].append(f"Attitude analysis error: {str(e)}")
        
        return issues

    def _analyze_power_anomalies(self, curr_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze power system anomalies."""
        issues = {'critical': [], 'warning': [], 'info': []}
        
        try:
            # Check battery voltage
            if 'Volt' in curr_df.columns:
                min_volt = curr_df['Volt'].min()
                if min_volt < 10.5:  # Low voltage threshold
                    issues['critical'].append(f"Low battery voltage: {min_volt:.1f}V")
                elif min_volt < 11.1:
                    issues['warning'].append(f"Battery voltage getting low: {min_volt:.1f}V")
            
            # Check current spikes
            if 'Curr' in curr_df.columns:
                max_current = curr_df['Curr'].max()
                mean_current = curr_df['Curr'].mean()
                if max_current > mean_current * 3:
                    issues['warning'].append(f"Current spikes: max {max_current:.1f}A (avg {mean_current:.1f}A)")
                    
        except Exception as e:
            issues['warning'].append(f"Power analysis error: {str(e)}")
        
        return issues

    def _analyze_general_anomalies(self, df: pd.DataFrame, system_name: str) -> Dict[str, List[str]]:
        """General anomaly detection for other systems."""
        issues = {'warning': [], 'info': []}
        
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 10:  # Need sufficient data
                        mean_val = values.mean()
                        std_val = values.std()
                        outliers = values[abs(values - mean_val) > 3 * std_val]
                        
                        if len(outliers) > len(values) * 0.05:  # > 5% outliers
                            issues['warning'].append(f"{system_name}.{col}: {len(outliers)} anomalous readings")
                        elif len(outliers) > 0:
                            issues['info'].append(f"{system_name}.{col}: {len(outliers)} minor outliers")
                            
        except Exception as e:
            issues['warning'].append(f"{system_name} analysis error: {str(e)}")
        
        return issues

    # Event detection helper methods
    def _detect_gps_loss_events(self, gps_df: pd.DataFrame) -> List[Dict]:
        """Detect GPS signal loss events with timestamps."""
        events = []
        
        try:
            if 'Status' in gps_df.columns:
                # Find transitions to poor GPS status
                prev_status = None
                for idx, row in gps_df.iterrows():
                    current_status = row['Status']
                    
                    # GPS signal lost (status < 3 means no 3D fix)
                    if prev_status is not None and prev_status >= 3 and current_status < 3:
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in gps_df.columns else "Unknown time"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"GPS signal lost (Status: {current_status})",
                            'severity': 'critical'
                        })
                    
                    # GPS signal recovered
                    elif prev_status is not None and prev_status < 3 and current_status >= 3:
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in gps_df.columns else "Unknown time"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"GPS signal recovered (Status: {current_status})",
                            'severity': 'info'
                        })
                    
                    prev_status = current_status
        except Exception as e:
            logger.error(f"Error detecting GPS events: {e}")
        
        return events

    def _detect_mode_changes(self, mode_df: pd.DataFrame) -> List[Dict]:
        """Detect flight mode changes."""
        events = []
        
        try:
            mode_names = {
                0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
                5: "LOITER", 6: "RTL", 7: "CIRCLE", 8: "POSITION", 9: "LAND",
                10: "AUTOTUNE", 11: "POSHOLD", 12: "BRAKE", 13: "THROW",
                14: "AVOID_ADSB", 15: "GUIDED_NOGPS", 16: "SMART_RTL",
                17: "FLOWHOLD", 18: "FOLLOW", 19: "ZIGZAG", 20: "SYSTEMID",
                21: "AUTOROTATE", 22: "AUTO_RTL"
            }
            
            for idx, row in mode_df.iterrows():
                mode_num = row.get('Mode', 0)
                mode_name = mode_names.get(mode_num, f"MODE_{mode_num}")
                time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in mode_df.columns else "Unknown time"
                
                events.append({
                    'timestamp': row.get('timestamp', 0),
                    'time_str': time_str,
                    'description': f"Mode changed to {mode_name}",
                    'severity': 'info'
                })
        except Exception as e:
            logger.error(f"Error detecting mode changes: {e}")
        
        return events

    def _detect_critical_alerts(self, msg_df: pd.DataFrame) -> List[Dict]:
        """Detect critical alert messages."""
        events = []
        
        try:
            critical_keywords = ['ERROR', 'CRITICAL', 'FAIL', 'EMERGENCY', 'LOST', 'WARNING']
            
            for idx, row in msg_df.iterrows():
                if 'Message' in row:
                    message = str(row['Message']).upper()
                    if any(keyword in message for keyword in critical_keywords):
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in msg_df.columns else "Unknown time"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Alert: {row['Message']}",
                            'severity': 'warning'
                        })
        except Exception as e:
            logger.error(f"Error detecting critical alerts: {e}")
        
        return events

    def _detect_power_issues(self, curr_df: pd.DataFrame) -> List[Dict]:
        """Detect power-related issues."""
        events = []
        
        try:
            if 'Volt' in curr_df.columns:
                low_voltage_threshold = 10.5
                critical_voltage_threshold = 10.0
                
                for idx, row in curr_df.iterrows():
                    voltage = row['Volt']
                    time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in curr_df.columns else "Unknown time"
                    
                    if voltage < critical_voltage_threshold:
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Critical low voltage: {voltage:.1f}V",
                            'severity': 'critical'
                        })
                    elif voltage < low_voltage_threshold:
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Low voltage warning: {voltage:.1f}V",
                            'severity': 'warning'
                        })
        except Exception as e:
            logger.error(f"Error detecting power issues: {e}")
        
        return events

    def _detect_attitude_problems(self, att_df: pd.DataFrame) -> List[Dict]:
        """Detect attitude control problems."""
        events = []
        
        try:
            for idx, row in att_df.iterrows():
                time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in att_df.columns else "Unknown time"
                
                # Check for extreme angles
                if 'Roll' in row and abs(row['Roll']) > 45:
                    events.append({
                        'timestamp': row.get('timestamp', 0),
                        'time_str': time_str,
                        'description': f"Extreme roll angle: {row['Roll']:.1f}°",
                        'severity': 'critical'
                    })
                
                if 'Pitch' in row and abs(row['Pitch']) > 45:
                    events.append({
                        'timestamp': row.get('timestamp', 0),
                        'time_str': time_str,
                        'description': f"Extreme pitch angle: {row['Pitch']:.1f}°",
                        'severity': 'critical'
                    })
        except Exception as e:
            logger.error(f"Error detecting attitude problems: {e}")
        
        return events

    # Timeline helper methods
    def _get_gps_timeline_events(self, gps_df: pd.DataFrame) -> List[Dict]:
        """Get GPS-related timeline events."""
        events = []
        
        try:
            # GPS status changes
            if 'Status' in gps_df.columns:
                prev_status = None
                for idx, row in gps_df.iterrows():
                    current_status = row['Status']
                    if prev_status is not None and prev_status != current_status:
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in gps_df.columns else "Unknown"
                        status_desc = "3D Fix" if current_status >= 3 else "No Fix"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"GPS status: {status_desc}",
                            'category': 'gps'
                        })
                    prev_status = current_status
        except Exception as e:
            logger.error(f"Error getting GPS timeline events: {e}")
        
        return events

    def _get_mode_timeline_events(self, mode_df: pd.DataFrame) -> List[Dict]:
        """Get mode change timeline events."""
        events = []
        
        try:
            mode_names = {
                0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
                5: "LOITER", 6: "RTL", 7: "CIRCLE", 8: "POSITION", 9: "LAND"
            }
            
            for idx, row in mode_df.iterrows():
                mode_num = row.get('Mode', 0)
                mode_name = mode_names.get(mode_num, f"MODE_{mode_num}")
                time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in mode_df.columns else "Unknown"
                
                events.append({
                    'timestamp': row.get('timestamp', 0),
                    'time_str': time_str,
                    'description': f"Mode: {mode_name}",
                    'category': 'mode'
                })
        except Exception as e:
            logger.error(f"Error getting mode timeline events: {e}")
        
        return events

    def _get_power_timeline_events(self, curr_df: pd.DataFrame) -> List[Dict]:
        """Get power-related timeline events."""
        events = []
        
        try:
            if 'Volt' in curr_df.columns:
                low_voltage_threshold = 11.0
                prev_low = False
                
                for idx, row in curr_df.iterrows():
                    voltage = row['Volt']
                    current_low = voltage < low_voltage_threshold
                    
                    if current_low and not prev_low:
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in curr_df.columns else "Unknown"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Low voltage: {voltage:.1f}V",
                            'category': 'power'
                        })
                    
                    prev_low = current_low
        except Exception as e:
            logger.error(f"Error getting power timeline events: {e}")
        
        return events

    def _get_message_timeline_events(self, msg_df: pd.DataFrame) -> List[Dict]:
        """Get message timeline events."""
        events = []
        
        try:
            important_keywords = ['ERROR', 'CRITICAL', 'FAIL', 'WARNING', 'ARMED', 'DISARMED']
            
            for idx, row in msg_df.iterrows():
                if 'Message' in row:
                    message = str(row['Message'])
                    if any(keyword in message.upper() for keyword in important_keywords):
                        time_str = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in msg_df.columns else "Unknown"
                        events.append({
                            'timestamp': row.get('timestamp', 0),
                            'time_str': time_str,
                            'description': f"Message: {message}",
                            'category': 'message'
                        })
        except Exception as e:
            logger.error(f"Error getting message timeline events: {e}")
        
        return events

    def _analyze_single_phase(self, session: V2ConversationSession, phase: str, metrics: List[str]) -> str:
        """Analyze a single flight phase."""
        try:
            phase_data = self._identify_flight_phase(session, phase)
            
            if not phase_data['data']:
                return f"❌ {phase.upper()} phase: No data available"
            
            analysis = f"✈️ {phase.upper()} PHASE:\n"
            analysis += f"Duration: {phase_data['duration']:.1f} seconds\n"
            analysis += f"Data points: {len(phase_data['data'])}\n"
            
            # Analyze specific metrics for this phase
            for metric in metrics:
                metric_analysis = self._analyze_phase_metric(session, phase_data, metric)
                if metric_analysis:
                    analysis += f"\n{metric.upper()}:\n{metric_analysis}\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing {phase} phase: {str(e)}"

    def _identify_flight_phase(self, session: V2ConversationSession, phase: str) -> Dict[str, Any]:
        """Identify and extract data for a specific flight phase."""
        phase_data = {'data': [], 'duration': 0, 'start_time': None, 'end_time': None}
        
        try:
            # Simple phase identification based on altitude patterns
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                
                if 'Alt' in gps_df.columns and 'timestamp' in gps_df.columns:
                    if phase == 'takeoff':
                        # Find initial altitude increase
                        start_alt = gps_df['Alt'].iloc[0]
                        takeoff_data = gps_df[gps_df['Alt'] < start_alt + 50]  # First 50m
                        phase_data['data'] = takeoff_data
                        
                    elif phase == 'cruise':
                        # Find stable altitude period
                        median_alt = gps_df['Alt'].median()
                        cruise_data = gps_df[abs(gps_df['Alt'] - median_alt) < 20]  # Within 20m of median
                        phase_data['data'] = cruise_data
                        
                    elif phase == 'landing':
                        # Find final altitude decrease
                        end_alt = gps_df['Alt'].iloc[-1]
                        landing_data = gps_df[gps_df['Alt'] < end_alt + 50]  # Last 50m
                        phase_data['data'] = landing_data.tail(100)  # Last 100 points
                    
                    # Calculate duration
                    if len(phase_data['data']) > 0:
                        timestamps = phase_data['data']['timestamp']
                        phase_data['duration'] = (timestamps.max() - timestamps.min()).total_seconds()
                        phase_data['start_time'] = timestamps.min()
                        phase_data['end_time'] = timestamps.max()
                        
        except Exception as e:
            logger.error(f"Error identifying {phase} phase: {e}")
        
        return phase_data

    def _analyze_phase_metric(self, session: V2ConversationSession, phase_data: Dict, metric: str) -> str:
        """Analyze a specific metric during a flight phase."""
        try:
            if not phase_data['data'] or len(phase_data['data']) == 0:
                return "No data available"
            
            df = phase_data['data']
            
            if metric == 'altitude':
                if 'Alt' in df.columns:
                    alt_change = df['Alt'].iloc[-1] - df['Alt'].iloc[0]
                    max_alt = df['Alt'].max()
                    min_alt = df['Alt'].min()
                    return f"  Altitude change: {alt_change:.1f}m\n  Range: {min_alt:.1f}m to {max_alt:.1f}m"
            
            elif metric == 'speed':
                if 'Spd' in df.columns:
                    avg_speed = df['Spd'].mean()
                    max_speed = df['Spd'].max()
                    return f"  Average speed: {avg_speed:.1f} m/s\n  Maximum speed: {max_speed:.1f} m/s"
            
            elif metric == 'power':
                # Need to get power data for the same time period
                if 'CURR' in session.dataframes:
                    curr_df = session.dataframes['CURR']
                    if 'timestamp' in curr_df.columns and phase_data['start_time'] and phase_data['end_time']:
                        phase_power = curr_df[
                            (curr_df['timestamp'] >= phase_data['start_time']) & 
                            (curr_df['timestamp'] <= phase_data['end_time'])
                        ]
                        if len(phase_power) > 0 and 'Curr' in phase_power.columns:
                            avg_current = phase_power['Curr'].mean()
                            max_current = phase_power['Curr'].max()
                            return f"  Average current: {avg_current:.1f}A\n  Peak current: {max_current:.1f}A"
            
            elif metric == 'stability':
                stability_info = []
                if 'Roll' in df.columns:
                    roll_std = df['Roll'].std()
                    stability_info.append(f"Roll stability: {roll_std:.1f}° std")
                if 'Pitch' in df.columns:
                    pitch_std = df['Pitch'].std()
                    stability_info.append(f"Pitch stability: {pitch_std:.1f}° std")
                
                return "\n  ".join(stability_info) if stability_info else "No stability data"
            
            return f"Metric '{metric}' analysis not available"
            
        except Exception as e:
            return f"Error analyzing {metric}: {str(e)}" 