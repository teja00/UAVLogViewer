"""
Anomaly detection for UAV flight data.
"""

import logging
import pandas as pd
from typing import Dict, List
from models import V2ConversationSession

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in UAV flight data."""

    def detect_anomalies(self, session: V2ConversationSession, focus_areas: List[str]) -> str:
        """Enhanced anomaly detection with temporal analysis and severity assessment."""
        critical_issues = []
        warning_issues = []
        info_issues = []
        
        try:
            for area in focus_areas:
                if area.upper() in session.dataframes:
                    df = session.dataframes[area.upper()]
                    
                    # GPS-specific analysis
                    if area.upper() == 'GPS':
                        issues = self._analyze_gps_anomalies(df)
                        critical_issues.extend(issues['critical'])
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
                    
                    # Attitude control analysis
                    elif area.upper() == 'ATT':
                        issues = self._analyze_attitude_anomalies(df)
                        critical_issues.extend(issues['critical'])
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
                    
                    # Power system analysis
                    elif area.upper() == 'CURR':
                        issues = self._analyze_power_anomalies(df)
                        critical_issues.extend(issues['critical'])
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
                    
                    # General analysis for other systems
                    else:
                        issues = self._analyze_general_anomalies(df, area.upper())
                        warning_issues.extend(issues['warning'])
                        info_issues.extend(issues['info'])
            
            # Format response with prioritization
            response_parts = []
            
            if critical_issues:
                response_parts.append("CRITICAL ISSUES:")
                for issue in critical_issues:
                    response_parts.append(f"  • {issue}")
            
            if warning_issues:
                response_parts.append("\nWARNING ISSUES:")
                for issue in warning_issues:
                    response_parts.append(f"  • {issue}")
            
            if info_issues:
                response_parts.append("\nINFO:")
                for issue in info_issues:
                    response_parts.append(f"  • {issue}")
            
            if not (critical_issues or warning_issues or info_issues):
                return "No significant anomalies detected in the analyzed areas."
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in enhanced anomaly detection: {e}")
            return f"Error analyzing anomalies: {str(e)}"

    def _analyze_gps_anomalies(self, gps_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detailed GPS-specific anomaly analysis."""
        issues = {'critical': [], 'warning': [], 'info': []}
        
        try:
            # Check for GPS signal loss
            if 'Status' in gps_df.columns:
                signal_loss_points = gps_df[gps_df['Status'] < 3]
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
                large_jumps = alt_diff[alt_diff > 50]
                if len(large_jumps) > 0:
                    max_jump = alt_diff.max()
                    issues['warning'].append(f"GPS altitude jumps detected: max {max_jump:.1f}m")
            
            # Check ground speed anomalies
            if 'Spd' in gps_df.columns:
                high_speed = gps_df[gps_df['Spd'] > 50]
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
                    if max_angle > 45:
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
                if min_volt < 10.5:
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
                    if len(values) > 10:
                        mean_val = values.mean()
                        std_val = values.std()
                        outliers = values[abs(values - mean_val) > 3 * std_val]
                        
                        if len(outliers) > len(values) * 0.05:
                            issues['warning'].append(f"{system_name}.{col}: {len(outliers)} anomalous readings")
                        elif len(outliers) > 0:
                            issues['info'].append(f"{system_name}.{col}: {len(outliers)} minor outliers")
                            
        except Exception as e:
            issues['warning'].append(f"{system_name} analysis error: {str(e)}")
        
        return issues 