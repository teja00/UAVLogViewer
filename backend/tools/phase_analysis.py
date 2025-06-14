"""
Flight phase analysis for UAV data.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from models import V2ConversationSession

logger = logging.getLogger(__name__)


class PhaseAnalyzer:
    """Analyzes flight phases with detailed metrics."""

    def analyze_phase(self, session: V2ConversationSession, phase: str, metrics: List[str]) -> str:
        """Analyze specific flight phase with detailed metrics."""
        try:
            # Identify the flight phase boundaries
            phase_data = self._identify_flight_phase(session, phase)
            
            if not phase_data or not phase_data.get('found', False):
                return f"Unable to identify {phase} phase in the flight data. Available dataframes: {list(session.dataframes.keys())}"
            
            # Analyze the specified metrics for this phase
            results = []
            results.append(f"=== {phase.upper()} PHASE ANALYSIS ===")
            results.append(f"Duration: {phase_data.get('duration', 'Unknown')}")
            results.append(f"Time range: {phase_data.get('start_time', 'Unknown')} to {phase_data.get('end_time', 'Unknown')}")
            
            if metrics:
                results.append("\nMETRIC ANALYSIS:")
                for metric in metrics:
                    metric_result = self._analyze_phase_metric(session, phase_data, metric)
                    results.append(f"• {metric.title()}: {metric_result}")
            else:
                # Provide default comprehensive analysis
                results.append("\nCOMPREHENSIVE ANALYSIS:")
                default_metrics = ['altitude', 'speed', 'power', 'stability']
                for metric in default_metrics:
                    metric_result = self._analyze_phase_metric(session, phase_data, metric)
                    if metric_result != "No data available":
                        results.append(f"• {metric.title()}: {metric_result}")
            
            # Add phase-specific insights
            insights = self._get_phase_insights(session, phase, phase_data)
            if insights:
                results.append("\nINSIGHTS:")
                for insight in insights:
                    results.append(f"• {insight}")
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error analyzing flight phase: {e}")
            return f"Error analyzing {phase} phase: {str(e)}"

    def _identify_flight_phase(self, session: V2ConversationSession, phase: str) -> Dict[str, Any]:
        """Identify flight phase boundaries based on available data."""
        phase_info = {'found': False, 'start_idx': None, 'end_idx': None}
        
        try:
            if phase.lower() == 'all':
                # Return entire flight data
                return {
                    'found': True,
                    'start_idx': 0,
                    'end_idx': -1,
                    'duration': 'Full flight',
                    'start_time': 'Flight start',
                    'end_time': 'Flight end',
                    'phase_type': 'complete_flight'
                }
            
            # Try to identify phases using different approaches
            phase_info = self._identify_phase_by_altitude(session, phase)
            
            if not phase_info['found']:
                phase_info = self._identify_phase_by_mode(session, phase)
            
            if not phase_info['found']:
                phase_info = self._identify_phase_by_speed(session, phase)
            
            # If still not found, use heuristics
            if not phase_info['found']:
                phase_info = self._identify_phase_by_heuristics(session, phase)
            
            return phase_info
            
        except Exception as e:
            logger.error(f"Error identifying flight phase: {e}")
            return {'found': False}

    def _identify_phase_by_altitude(self, session: V2ConversationSession, phase: str) -> Dict[str, Any]:
        """Identify flight phase using altitude data."""
        phase_info = {'found': False}
        
        try:
            # Try different altitude sources
            altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2', 'POS']
            alt_df = None
            alt_col = None
            
            for source in altitude_sources:
                if source in session.dataframes:
                    df = session.dataframes[source]
                    if 'Alt' in df.columns and not df['Alt'].empty:
                        alt_df = df
                        alt_col = 'Alt'
                        break
            
            if alt_df is None or alt_col is None:
                return phase_info
            
            altitudes = alt_df[alt_col].dropna()
            if len(altitudes) < 10:
                return phase_info
            
            # Calculate altitude statistics
            alt_min = altitudes.min()
            alt_max = altitudes.max()
            alt_range = alt_max - alt_min
            
            if phase.lower() == 'takeoff':
                # Takeoff: first part where altitude increases significantly
                takeoff_threshold = alt_min + (alt_range * 0.2)  # First 20% of altitude gain
                takeoff_data = altitudes[altitudes <= takeoff_threshold]
                if len(takeoff_data) > 0:
                    end_idx = len(takeoff_data)
                    phase_info = {
                        'found': True,
                        'start_idx': 0,
                        'end_idx': end_idx,
                        'duration': f'{end_idx} data points',
                        'start_time': 'Flight start',
                        'end_time': f'Altitude reached {takeoff_threshold:.1f}m',
                        'phase_type': 'takeoff',
                        'altitude_range': f'{alt_min:.1f}m to {takeoff_threshold:.1f}m'
                    }
            
            elif phase.lower() == 'landing':
                # Landing: last part where altitude decreases significantly
                landing_threshold = alt_min + (alt_range * 0.2)  # Last 20% of descent
                landing_start_alt = alt_max - (alt_range * 0.3)  # Start descent detection
                landing_data = altitudes[altitudes >= landing_start_alt]
                if len(landing_data) > 0:
                    start_idx = len(altitudes) - len(landing_data)
                    phase_info = {
                        'found': True,
                        'start_idx': start_idx,
                        'end_idx': -1,
                        'duration': f'{len(landing_data)} data points',
                        'start_time': f'Descent from {landing_start_alt:.1f}m',
                        'end_time': 'Flight end',
                        'phase_type': 'landing',
                        'altitude_range': f'{landing_start_alt:.1f}m to {alt_min:.1f}m'
                    }
            
            elif phase.lower() == 'cruise':
                # Cruise: middle part with relatively stable altitude
                cruise_min = alt_min + (alt_range * 0.3)
                cruise_max = alt_max - (alt_range * 0.3)
                cruise_data = altitudes[(altitudes >= cruise_min) & (altitudes <= cruise_max)]
                if len(cruise_data) > 0:
                    # Find the continuous cruise section
                    cruise_indices = altitudes[(altitudes >= cruise_min) & (altitudes <= cruise_max)].index
                    if len(cruise_indices) > 0:
                        start_idx = cruise_indices[0]
                        end_idx = cruise_indices[-1]
                        phase_info = {
                            'found': True,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'duration': f'{len(cruise_data)} data points',
                            'start_time': f'Cruise altitude reached',
                            'end_time': f'Cruise phase end',
                            'phase_type': 'cruise',
                            'altitude_range': f'{cruise_min:.1f}m to {cruise_max:.1f}m'
                        }
            
        except Exception as e:
            logger.error(f"Error identifying phase by altitude: {e}")
        
        return phase_info

    def _identify_phase_by_mode(self, session: V2ConversationSession, phase: str) -> Dict[str, Any]:
        """Identify flight phase using mode changes."""
        phase_info = {'found': False}
        
        try:
            if 'MODE' not in session.dataframes:
                return phase_info
            
            mode_df = session.dataframes['MODE']
            if mode_df.empty:
                return phase_info
            
            # Mode-based phase detection
            phase_modes = {
                'takeoff': ['GUIDED', 'AUTO', 'TAKEOFF'],
                'cruise': ['AUTO', 'GUIDED', 'LOITER'],
                'landing': ['LAND', 'RTL', 'GUIDED']
            }
            
            if phase.lower() in phase_modes:
                target_modes = phase_modes[phase.lower()]
                if 'asText' in mode_df.columns:
                    matching_modes = mode_df[mode_df['asText'].isin(target_modes)]
                    if not matching_modes.empty:
                        phase_info = {
                            'found': True,
                            'start_idx': 0,  # Simplified for mode-based detection
                            'end_idx': -1,
                            'duration': f'{len(matching_modes)} mode periods',
                            'start_time': 'Mode-based detection',
                            'end_time': 'Mode-based detection',
                            'phase_type': f'{phase}_mode_based',
                            'modes': matching_modes['asText'].tolist()
                        }
        
        except Exception as e:
            logger.error(f"Error identifying phase by mode: {e}")
        
        return phase_info

    def _identify_phase_by_speed(self, session: V2ConversationSession, phase: str) -> Dict[str, Any]:
        """Identify flight phase using speed data."""
        phase_info = {'found': False}
        
        try:
            if 'GPS' not in session.dataframes:
                return phase_info
            
            gps_df = session.dataframes['GPS']
            if 'Spd' not in gps_df.columns or gps_df['Spd'].empty:
                return phase_info
            
            speeds = gps_df['Spd'].dropna()
            if len(speeds) < 10:
                return phase_info
            
            max_speed = speeds.max()
            avg_speed = speeds.mean()
            
            if phase.lower() == 'takeoff':
                # Takeoff: low to increasing speed
                takeoff_speeds = speeds[speeds <= avg_speed * 0.5]
                if len(takeoff_speeds) > 0:
                    phase_info = {
                        'found': True,
                        'start_idx': 0,
                        'end_idx': len(takeoff_speeds),
                        'duration': f'{len(takeoff_speeds)} data points',
                        'start_time': 'Low speed start',
                        'end_time': f'Speed reached {avg_speed * 0.5:.1f} m/s',
                        'phase_type': 'takeoff_speed_based',
                        'speed_range': f'0 to {avg_speed * 0.5:.1f} m/s'
                    }
            
            elif phase.lower() == 'cruise':
                # Cruise: stable medium to high speed
                cruise_speeds = speeds[(speeds >= avg_speed * 0.7) & (speeds <= max_speed * 0.9)]
                if len(cruise_speeds) > 0:
                    phase_info = {
                        'found': True,
                        'start_idx': 0,  # Simplified
                        'end_idx': -1,
                        'duration': f'{len(cruise_speeds)} data points',
                        'start_time': 'Cruise speed reached',
                        'end_time': 'Cruise speed maintained',
                        'phase_type': 'cruise_speed_based',
                        'speed_range': f'{avg_speed * 0.7:.1f} to {max_speed * 0.9:.1f} m/s'
                    }
        
        except Exception as e:
            logger.error(f"Error identifying phase by speed: {e}")
        
        return phase_info

    def _identify_phase_by_heuristics(self, session: V2ConversationSession, phase: str) -> Dict[str, Any]:
        """Fallback heuristic-based phase identification."""
        try:
            # Get total flight duration estimate
            total_records = sum(len(df) for df in session.dataframes.values()) / len(session.dataframes)
            
            if phase.lower() == 'takeoff':
                return {
                    'found': True,
                    'start_idx': 0,
                    'end_idx': int(total_records * 0.15),  # First 15%
                    'duration': 'Estimated first 15% of flight',
                    'start_time': 'Flight start',
                    'end_time': 'Estimated takeoff completion',
                    'phase_type': 'heuristic_takeoff'
                }
            
            elif phase.lower() == 'landing':
                return {
                    'found': True,
                    'start_idx': int(total_records * 0.85),  # Last 15%
                    'end_idx': -1,
                    'duration': 'Estimated last 15% of flight',
                    'start_time': 'Estimated landing start',
                    'end_time': 'Flight end', 
                    'phase_type': 'heuristic_landing'
                }
            
            elif phase.lower() == 'cruise':
                return {
                    'found': True,
                    'start_idx': int(total_records * 0.15),
                    'end_idx': int(total_records * 0.85),
                    'duration': 'Estimated middle 70% of flight',
                    'start_time': 'Estimated cruise start',
                    'end_time': 'Estimated cruise end',
                    'phase_type': 'heuristic_cruise'
                }
        
        except Exception as e:
            logger.error(f"Error in heuristic phase identification: {e}")
        
        return {'found': False}

    def _analyze_phase_metric(self, session: V2ConversationSession, phase_data: Dict, metric: str) -> str:
        """Analyze a specific metric for the identified phase."""
        try:
            if metric.lower() == 'altitude':
                return self._analyze_altitude_metric(session, phase_data)
            elif metric.lower() == 'speed':
                return self._analyze_speed_metric(session, phase_data)
            elif metric.lower() == 'power':
                return self._analyze_power_metric(session, phase_data)
            elif metric.lower() == 'stability':
                return self._analyze_stability_metric(session, phase_data)
            else:
                return f"Unknown metric: {metric}"
                
        except Exception as e:
            logger.error(f"Error analyzing phase metric {metric}: {e}")
            return f"Error analyzing {metric}: {str(e)}"

    def _analyze_altitude_metric(self, session: V2ConversationSession, phase_data: Dict) -> str:
        """Analyze altitude for the phase."""
        try:
            # Find altitude data
            altitude_sources = ['GPS', 'BARO', 'CTUN', 'AHR2']
            for source in altitude_sources:
                if source in session.dataframes:
                    df = session.dataframes[source]
                    if 'Alt' in df.columns and not df['Alt'].empty:
                        altitudes = df['Alt'].dropna()
                        
                        # Apply phase filtering if indices are available
                        start_idx = phase_data.get('start_idx', 0)
                        end_idx = phase_data.get('end_idx', -1)
                        
                        if start_idx is not None and end_idx is not None:
                            if end_idx == -1:
                                phase_altitudes = altitudes.iloc[start_idx:]
                            else:
                                phase_altitudes = altitudes.iloc[start_idx:end_idx]
                        else:
                            phase_altitudes = altitudes
                        
                        if len(phase_altitudes) > 0:
                            min_alt = phase_altitudes.min()
                            max_alt = phase_altitudes.max()
                            avg_alt = phase_altitudes.mean()
                            alt_change = max_alt - min_alt
                            
                            return f"Range: {min_alt:.1f}m to {max_alt:.1f}m, Average: {avg_alt:.1f}m, Change: {alt_change:.1f}m ({source})"
            
            return "No altitude data available"
            
        except Exception as e:
            logger.error(f"Error analyzing altitude metric: {e}")
            return f"Error: {str(e)}"

    def _analyze_speed_metric(self, session: V2ConversationSession, phase_data: Dict) -> str:
        """Analyze speed for the phase."""
        try:
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'Spd' in gps_df.columns and not gps_df['Spd'].empty:
                    speeds = gps_df['Spd'].dropna()
                    
                    # Apply phase filtering
                    start_idx = phase_data.get('start_idx', 0)
                    end_idx = phase_data.get('end_idx', -1)
                    
                    if start_idx is not None and end_idx is not None:
                        if end_idx == -1:
                            phase_speeds = speeds.iloc[start_idx:]
                        else:
                            phase_speeds = speeds.iloc[start_idx:end_idx]
                    else:
                        phase_speeds = speeds
                    
                    if len(phase_speeds) > 0:
                        min_spd = phase_speeds.min()
                        max_spd = phase_speeds.max()
                        avg_spd = phase_speeds.mean()
                        
                        return f"Range: {min_spd:.1f} to {max_spd:.1f} m/s, Average: {avg_spd:.1f} m/s"
            
            return "No speed data available"
            
        except Exception as e:
            logger.error(f"Error analyzing speed metric: {e}")
            return f"Error: {str(e)}"

    def _analyze_power_metric(self, session: V2ConversationSession, phase_data: Dict) -> str:
        """Analyze power consumption for the phase."""
        try:
            if 'CURR' in session.dataframes:
                curr_df = session.dataframes['CURR']
                
                results = []
                if 'Volt' in curr_df.columns and not curr_df['Volt'].empty:
                    voltages = curr_df['Volt'].dropna()
                    
                    # Apply phase filtering
                    start_idx = phase_data.get('start_idx', 0)
                    end_idx = phase_data.get('end_idx', -1)
                    
                    if start_idx is not None and end_idx is not None:
                        if end_idx == -1:
                            phase_voltages = voltages.iloc[start_idx:]
                        else:
                            phase_voltages = voltages.iloc[start_idx:end_idx]
                    else:
                        phase_voltages = voltages
                    
                    if len(phase_voltages) > 0:
                        min_volt = phase_voltages.min()
                        max_volt = phase_voltages.max()
                        avg_volt = phase_voltages.mean()
                        results.append(f"Voltage: {min_volt:.1f} to {max_volt:.1f}V (avg: {avg_volt:.1f}V)")
                
                if 'Curr' in curr_df.columns and not curr_df['Curr'].empty:
                    currents = curr_df['Curr'].dropna()
                    
                    # Apply same phase filtering to current
                    start_idx = phase_data.get('start_idx', 0)
                    end_idx = phase_data.get('end_idx', -1)
                    
                    if start_idx is not None and end_idx is not None:
                        if end_idx == -1:
                            phase_currents = currents.iloc[start_idx:]
                        else:
                            phase_currents = currents.iloc[start_idx:end_idx]
                    else:
                        phase_currents = currents
                    
                    if len(phase_currents) > 0:
                        avg_curr = phase_currents.mean()
                        max_curr = phase_currents.max()
                        results.append(f"Current: {avg_curr:.1f}A avg, {max_curr:.1f}A max")
                
                return ", ".join(results) if results else "No power data available"
            
            return "No power data available"
            
        except Exception as e:
            logger.error(f"Error analyzing power metric: {e}")
            return f"Error: {str(e)}"

    def _analyze_stability_metric(self, session: V2ConversationSession, phase_data: Dict) -> str:
        """Analyze flight stability for the phase."""
        try:
            stability_results = []
            
            # Check attitude stability
            if 'ATT' in session.dataframes:
                att_df = session.dataframes['ATT']
                
                for angle in ['Roll', 'Pitch']:
                    if angle in att_df.columns:
                        angles = att_df[angle].dropna()
                        
                        # Apply phase filtering
                        start_idx = phase_data.get('start_idx', 0)
                        end_idx = phase_data.get('end_idx', -1)
                        
                        if start_idx is not None and end_idx is not None:
                            if end_idx == -1:
                                phase_angles = angles.iloc[start_idx:]
                            else:
                                phase_angles = angles.iloc[start_idx:end_idx]
                        else:
                            phase_angles = angles
                        
                        if len(phase_angles) > 0:
                            std_angle = phase_angles.std()
                            max_angle = phase_angles.abs().max()
                            stability_results.append(f"{angle}: ±{max_angle:.1f}° max, {std_angle:.1f}° std")
            
            # Check vibration if available
            if 'VIBE' in session.dataframes:
                vibe_df = session.dataframes['VIBE']
                vibe_cols = ['VibeX', 'VibeY', 'VibeZ']
                
                for col in vibe_cols:
                    if col in vibe_df.columns:
                        vibes = vibe_df[col].dropna()
                        if len(vibes) > 0:
                            avg_vibe = vibes.mean()
                            stability_results.append(f"{col}: {avg_vibe:.1f} avg")
                        break  # Just show one vibration metric
            
            return ", ".join(stability_results) if stability_results else "No stability data available"
            
        except Exception as e:
            logger.error(f"Error analyzing stability metric: {e}")
            return f"Error: {str(e)}"

    def _get_phase_insights(self, session: V2ConversationSession, phase: str, phase_data: Dict) -> List[str]:
        """Generate phase-specific insights."""
        insights = []
        
        try:
            phase_type = phase_data.get('phase_type', phase)
            
            if 'takeoff' in phase_type.lower():
                insights.append("Takeoff phase analysis based on altitude and speed changes")
                if 'altitude_range' in phase_data:
                    insights.append(f"Altitude gained during takeoff: {phase_data['altitude_range']}")
            
            elif 'landing' in phase_type.lower():
                insights.append("Landing phase identified by descent pattern")
                if 'altitude_range' in phase_data:
                    insights.append(f"Descent range during landing: {phase_data['altitude_range']}")
            
            elif 'cruise' in phase_type.lower():
                insights.append("Cruise phase represents stable flight conditions")
                
            if 'heuristic' in phase_type.lower():
                insights.append("Phase boundaries estimated using flight duration heuristics")
                
        except Exception as e:
            logger.error(f"Error generating phase insights: {e}")
        
        return insights 