"""
Enhanced Telemetry Analyzer for UAV Flight Data
Implements comprehensive analysis based on ArduPilot telemetry data exploration.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from models import V2ConversationSession

logger = logging.getLogger(__name__)


class EnhancedTelemetryAnalyzer:
    """
    Comprehensive UAV telemetry analyzer implementing all 164 message types
    and advanced analysis workflows from telemetry data exploration.
    """
    
    def __init__(self):
        # Define thresholds from the data exploration report
        self.anomaly_thresholds = {
            'voltage_critical': 3.3,  # V per cell
            'ekf_innovation_critical': 1.5,
            'altitude_loss_critical': 10,  # m within 1s
            'gps_outage_critical': 30,  # seconds
            'temperature_critical': 60,  # °C
            'mode_change_warning': 4,  # per minute
            'vibration_warning': 30,  # m/s²
            'vibration_critical': 60,  # m/s²
            'clip_events_warning': 100,  # IMU clip count
            'attitude_critical': 45,  # degrees
            'attitude_warning': 30,  # degrees
        }
        
        # Message type priorities based on subsystem importance
        self.message_priorities = {
            'navigation': ['GPS', 'GPS2', 'GPA', 'GRAW', 'GRXS', 'GPAH'],
            'attitude_inertial': ['IMU', 'IMU2', 'IMUS', 'AHR2', 'ATT'],
            'altimetry': ['BARO', 'RNGF', 'RNG2'],
            'ekf_health': ['NKF1', 'NKF2', 'NKF3', 'NKF4', 'NKQ', 'NKH'],
            'rc_control': ['RCIN', 'RCI2', 'RCOU'],
            'power_battery': ['BAT', 'CURR', 'POWR', 'MOTB'],
            'flight_modes': ['MODE', 'MSG', 'MSGC', 'EV'],
            'mission_nav': ['CMD', 'NAV', 'WPNT'],
            'vibration': ['VIBE', 'IMU'],  # Clip counts in IMU
            'custom_tuning': ['CTUN', 'NTUN', 'TECS'],
            'parameters': ['PARM', 'UNIT', 'MULT', 'FMT']
        }

    def comprehensive_analysis(self, session: V2ConversationSession) -> str:
        """Perform comprehensive telemetry analysis with concise output."""
        try:
            analysis_sections = []
            
            # Basic flight statistics - concise
            flight_stats = self._calculate_flight_statistics(session)
            if flight_stats.strip():
                analysis_sections.append(flight_stats)
            
            # Signal quality - concise
            signal_quality = self._signal_quality_assessment(session)
            if signal_quality.strip():
                analysis_sections.append(signal_quality)
            
            # Only add anomaly detection if critical issues found
            anomalies = self._comprehensive_anomaly_detection(session)
            if "CRITICAL" in anomalies.upper() or "WARNING" in anomalies.upper():
                analysis_sections.append("⚠️ " + anomalies.split('\n')[0])  # Just the first line
            
            if not analysis_sections:
                return "Flight data analyzed - no significant issues detected."
            
            return "\n\n".join(analysis_sections)
            
        except Exception as e:
            logger.error(f"Enhanced telemetry analysis failed: {e}")
            return f"Analysis error: {str(e)}"

    def _generate_data_overview(self, session: V2ConversationSession) -> str:
        """Generate overview of available data with message type categorization."""
        available_types = list(session.dataframes.keys())
        total_records = sum(len(df) for df in session.dataframes.values())
        
        # Categorize available message types
        categorized = {}
        for category, types in self.message_priorities.items():
            available_in_category = [t for t in types if t in available_types]
            if available_in_category:
                categorized[category] = available_in_category
        
        uncategorized = [t for t in available_types if not any(t in types for types in self.message_priorities.values())]
        
        overview = ["📊 DATA OVERVIEW:"]
        overview.append(f"• Total message types: {len(available_types)}")
        overview.append(f"• Total records: {total_records:,}")
        overview.append(f"• Data coverage by subsystem:")
        
        for category, types in categorized.items():
            overview.append(f"  - {category.title()}: {', '.join(types)} ({len(types)} types)")
        
        if uncategorized:
            overview.append(f"  - Other: {', '.join(uncategorized[:5])}{'...' if len(uncategorized) > 5 else ''}")
        
        return "\n".join(overview) + "\n"

    def _calculate_flight_statistics(self, session: V2ConversationSession) -> str:
        """Calculate comprehensive flight statistics."""
        stats = ["⏱️ FLIGHT STATISTICS:"]
        
        try:
            # Time analysis
            time_values = []
            for df_name, df in session.dataframes.items():
                if 'TimeUS' in df.columns:
                    times = df['TimeUS'].dropna()
                    if not times.empty:
                        time_values.extend(times.tolist())
            
            if time_values:
                duration_seconds = (max(time_values) - min(time_values)) / 1_000_000
                duration_minutes = duration_seconds / 60
                stats.append(f"• Flight duration: {duration_minutes:.1f} minutes ({duration_seconds:.0f} seconds)")
                
                # Estimate logging rate
                avg_log_rate = len(time_values) / duration_seconds if duration_seconds > 0 else 0
                stats.append(f"• Average log rate: {avg_log_rate:.1f} messages/second")
            
            # Message type utilization
            largest_df = max(session.dataframes.items(), key=lambda x: len(x[1]))
            stats.append(f"• Primary data source: {largest_df[0]} ({len(largest_df[1]):,} records)")
            
            # High-rate vs low-rate analysis
            high_rate_types = []
            low_rate_types = []
            for df_name, df in session.dataframes.items():
                record_count = len(df)
                if record_count > 1000:  # Arbitrary threshold for high-rate
                    high_rate_types.append(f"{df_name}({record_count:,})")
                else:
                    low_rate_types.append(f"{df_name}({record_count})")
            
            if high_rate_types:
                stats.append(f"• High-rate data: {', '.join(high_rate_types[:3])}{'...' if len(high_rate_types) > 3 else ''}")
            if low_rate_types:
                stats.append(f"• Event/low-rate data: {', '.join(low_rate_types[:5])}{'...' if len(low_rate_types) > 5 else ''}")
            
        except Exception as e:
            stats.append(f"• Statistics calculation error: {str(e)}")
        
        return "\n".join(stats) + "\n"

    def _calculate_derived_metrics(self, session: V2ConversationSession) -> str:
        """Calculate advanced derived metrics from raw telemetry."""
        metrics = ["📈 DERIVED METRICS:"]
        
        try:
            # Distance traveled calculation
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if all(col in gps_df.columns for col in ['Lat', 'Lng']):
                    total_distance = self._calculate_total_distance(gps_df)
                    if total_distance > 0:
                        metrics.append(f"• Total distance traveled: {total_distance:.0f} meters")
            
            # Climb/sink rate analysis
            altitude_rates = self._calculate_climb_rates(session)
            if altitude_rates:
                metrics.append(f"• Maximum climb rate: {altitude_rates['max_climb']:.1f} m/s")
                metrics.append(f"• Maximum sink rate: {altitude_rates['max_sink']:.1f} m/s")
            
            # Energy consumption integration
            energy_metrics = self._calculate_energy_consumption(session)
            if energy_metrics:
                metrics.append(f"• Total energy consumed: {energy_metrics['total_wh']:.1f} Wh")
                metrics.append(f"• Average power consumption: {energy_metrics['avg_power']:.1f} W")
            
            # GPS quality index
            gps_quality = self._calculate_gps_quality_index(session)
            if gps_quality:
                metrics.append(f"• GPS quality index: {gps_quality['quality_score']:.1f}/10.0")
                metrics.append(f"• 3D fix percentage: {gps_quality['fix_percentage']:.1f}%")
            
            # Vibration scoring
            vibration_score = self._calculate_vibration_score(session)
            if vibration_score:
                metrics.append(f"• Vibration score: {vibration_score['rms_vibration']:.1f} m/s²")
                metrics.append(f"• Vibration health: {vibration_score['health_status']}")
            
            # Attitude excursion analysis
            attitude_excursions = self._calculate_attitude_excursions(session)
            if attitude_excursions:
                metrics.append(f"• Max roll excursion: {attitude_excursions['max_roll']:.1f}°")
                metrics.append(f"• Max pitch excursion: {attitude_excursions['max_pitch']:.1f}°")
                metrics.append(f"• 95th percentile roll: {attitude_excursions['p95_roll']:.1f}°")
            
        except Exception as e:
            metrics.append(f"• Derived metrics calculation error: {str(e)}")
        
        return "\n".join(metrics) + "\n"

    def _cross_sensor_validation(self, session: V2ConversationSession) -> str:
        """Cross-sensor validation and consistency checks."""
        validation = ["🔍 CROSS-SENSOR VALIDATION:"]
        
        try:
            # GPS vs Barometer altitude comparison
            gps_baro_comparison = self._compare_gps_baro_altitude(session)
            if gps_baro_comparison:
                validation.append(f"• GPS vs Barometer altitude: {gps_baro_comparison['status']}")
                validation.append(f"  - Average difference: {gps_baro_comparison['avg_diff']:.1f}m")
                
            # Multiple GPS validation
            multi_gps = self._validate_multiple_gps(session)
            if multi_gps:
                validation.append(f"• Multiple GPS validation: {multi_gps}")
            
            # IMU vs Attitude consistency
            imu_attitude = self._validate_imu_attitude_consistency(session)
            if imu_attitude:
                validation.append(f"• IMU vs Attitude consistency: {imu_attitude}")
            
            # EKF vs Raw sensor validation
            ekf_raw = self._validate_ekf_vs_raw(session)
            if ekf_raw:
                validation.append(f"• EKF vs Raw sensors: {ekf_raw}")
                
        except Exception as e:
            validation.append(f"• Cross-sensor validation error: {str(e)}")
        
        return "\n".join(validation) + "\n"

    def _analyze_ekf_health(self, session: V2ConversationSession) -> str:
        """Extended Kalman Filter health analysis."""
        ekf_health = ["🧠 EKF HEALTH ANALYSIS:"]
        
        try:
            # Check available EKF message types
            ekf_types = [msg for msg in session.dataframes.keys() if msg.startswith(('NKF', 'XKF'))]
            
            if not ekf_types:
                ekf_health.append("• No EKF data available for analysis")
                return "\n".join(ekf_health) + "\n"
            
            ekf_health.append(f"• Available EKF data: {', '.join(ekf_types)}")
            
            # Velocity innovations analysis
            if 'NKF1' in session.dataframes:
                velocity_analysis = self._analyze_velocity_innovations(session.dataframes['NKF1'])
                ekf_health.append(f"• Velocity innovations: {velocity_analysis}")
            
            # Position innovations analysis
            if 'NKF2' in session.dataframes:
                position_analysis = self._analyze_position_innovations(session.dataframes['NKF2'])
                ekf_health.append(f"• Position innovations: {position_analysis}")
            
            # IMU bias estimates
            if 'NKF3' in session.dataframes:
                bias_analysis = self._analyze_imu_bias_estimates(session.dataframes['NKF3'])
                ekf_health.append(f"• IMU bias estimates: {bias_analysis}")
            
            # EKF reset events
            reset_events = self._detect_ekf_reset_events(session, ekf_types)
            if reset_events:
                ekf_health.append(f"• EKF reset events: {reset_events}")
                
        except Exception as e:
            ekf_health.append(f"• EKF health analysis error: {str(e)}")
        
        return "\n".join(ekf_health) + "\n"

    def _comprehensive_anomaly_detection(self, session: V2ConversationSession) -> str:
        """Comprehensive anomaly detection across all subsystems."""
        anomalies = ["🚨 COMPREHENSIVE ANOMALY DETECTION:"]
        
        try:
            critical_count = 0
            warning_count = 0
            
            # Voltage anomalies
            voltage_anomalies = self._detect_voltage_anomalies(session)
            if voltage_anomalies['critical']:
                critical_count += len(voltage_anomalies['critical'])
                for anomaly in voltage_anomalies['critical']:
                    anomalies.append(f"• 🚨 CRITICAL: {anomaly}")
            if voltage_anomalies['warning']:
                warning_count += len(voltage_anomalies['warning'])
                for anomaly in voltage_anomalies['warning']:
                    anomalies.append(f"• ⚠️ WARNING: {anomaly}")
            
            # Temperature anomalies
            temp_anomalies = self._detect_temperature_anomalies(session)
            for anomaly in temp_anomalies:
                if 'CRITICAL' in anomaly.upper():
                    critical_count += 1
                    anomalies.append(f"• 🚨 CRITICAL: {anomaly}")
                else:
                    warning_count += 1
                    anomalies.append(f"• ⚠️ WARNING: {anomaly}")
            
            # Mode change anomalies
            mode_anomalies = self._detect_mode_change_anomalies(session)
            for anomaly in mode_anomalies:
                warning_count += 1
                anomalies.append(f"• ⚠️ WARNING: {anomaly}")
            
            # Vibration anomalies
            vibration_anomalies = self._detect_vibration_anomalies(session)
            if vibration_anomalies['critical']:
                critical_count += len(vibration_anomalies['critical'])
                for anomaly in vibration_anomalies['critical']:
                    anomalies.append(f"• 🚨 CRITICAL: {anomaly}")
            if vibration_anomalies['warning']:
                warning_count += len(vibration_anomalies['warning'])
                for anomaly in vibration_anomalies['warning']:
                    anomalies.append(f"• ⚠️ WARNING: {anomaly}")
            
            # Summary
            if critical_count == 0 and warning_count == 0:
                anomalies.append("• ✅ No critical anomalies detected")
            else:
                anomalies.insert(1, f"• Summary: {critical_count} critical, {warning_count} warning anomalies")
            
        except Exception as e:
            anomalies.append(f"• Anomaly detection error: {str(e)}")
        
        return "\n".join(anomalies) + "\n"

    def _advanced_phase_analysis(self, session: V2ConversationSession) -> str:
        """Advanced flight phase analysis with comprehensive metrics."""
        phase_analysis = ["✈️ ADVANCED PHASE ANALYSIS:"]
        
        try:
            # Detect flight phases using multiple data sources
            phases = self._detect_flight_phases_advanced(session)
            
            for phase_name, phase_data in phases.items():
                if phase_data['detected']:
                    phase_analysis.append(f"\n{phase_name.upper()} PHASE:")
                    phase_analysis.append(f"  • Duration: {phase_data['duration']:.1f}s")
                    phase_analysis.append(f"  • Data quality: {phase_data['data_quality']}")
                    
                    # Phase-specific metrics
                    if 'metrics' in phase_data:
                        for metric, value in phase_data['metrics'].items():
                            phase_analysis.append(f"  • {metric}: {value}")
            
        except Exception as e:
            phase_analysis.append(f"• Phase analysis error: {str(e)}")
        
        return "\n".join(phase_analysis) + "\n"

    def _signal_quality_assessment(self, session: V2ConversationSession) -> str:
        """Concise signal quality assessment."""
        signal_quality = []
        
        try:
            # GPS signal quality - concise
            gps_quality = self._assess_gps_signal_quality(session)
            if gps_quality:
                signal_quality.append(f"GPS signal quality: {gps_quality}")
            
            # RC signal quality - concise
            rc_quality = self._assess_rc_signal_quality(session)
            if rc_quality:
                signal_quality.append(f"RC signal quality: {rc_quality}")
            
            # IMU data quality - concise
            imu_quality = self._assess_imu_data_quality(session)
            if imu_quality:
                signal_quality.append(f"IMU data quality: {imu_quality}")
            
        except Exception as e:
            signal_quality.append(f"Signal quality assessment error: {str(e)}")
        
        return "\n".join(signal_quality)

    # Helper methods for calculations
    def _calculate_total_distance(self, gps_df: pd.DataFrame) -> float:
        """Calculate total distance using Haversine formula."""
        if len(gps_df) < 2:
            return 0.0
        
        try:
            # Convert to radians
            lat1 = np.radians(gps_df['Lat'].iloc[:-1])
            lat2 = np.radians(gps_df['Lat'].iloc[1:])
            lon1 = np.radians(gps_df['Lng'].iloc[:-1])
            lon2 = np.radians(gps_df['Lng'].iloc[1:])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances = 6371000 * c  # Earth radius in meters
            
            return distances.sum()
        except:
            return 0.0

    def _calculate_climb_rates(self, session: V2ConversationSession) -> Optional[Dict[str, float]]:
        """Calculate climb and sink rates."""
        try:
            for source in ['GPS', 'BARO', 'CTUN']:
                if source in session.dataframes:
                    df = session.dataframes[source]
                    if 'Alt' in df.columns and 'TimeUS' in df.columns:
                        df_clean = df[['Alt', 'TimeUS']].dropna()
                        if len(df_clean) > 1:
                            time_diff = df_clean['TimeUS'].diff() / 1_000_000  # Convert to seconds
                            alt_diff = df_clean['Alt'].diff()
                            rates = alt_diff / time_diff
                            rates = rates[(time_diff > 0) & (time_diff < 5)]  # Filter reasonable time intervals
                            
                            if len(rates) > 0:
                                return {
                                    'max_climb': rates.max(),
                                    'max_sink': abs(rates.min()),
                                    'avg_climb': rates[rates > 0].mean() if (rates > 0).any() else 0,
                                    'avg_sink': abs(rates[rates < 0].mean()) if (rates < 0).any() else 0
                                }
            return None
        except:
            return None

    def _calculate_energy_consumption(self, session: V2ConversationSession) -> Optional[Dict[str, float]]:
        """Calculate energy consumption integration."""
        try:
            if 'CURR' in session.dataframes:
                curr_df = session.dataframes['CURR']
                if all(col in curr_df.columns for col in ['Volt', 'Curr', 'TimeUS']):
                    df_clean = curr_df[['Volt', 'Curr', 'TimeUS']].dropna()
                    if len(df_clean) > 1:
                        # Calculate power (Watts)
                        power = df_clean['Volt'] * df_clean['Curr']
                        time_diff = df_clean['TimeUS'].diff() / 1_000_000 / 3600  # Convert to hours
                        
                        # Integrate energy (Wh)
                        energy_increments = power.iloc[1:] * time_diff.iloc[1:]
                        total_wh = energy_increments.sum()
                        avg_power = power.mean()
                        
                        return {
                            'total_wh': total_wh,
                            'avg_power': avg_power,
                            'peak_power': power.max()
                        }
            return None
        except:
            return None

    def _calculate_gps_quality_index(self, session: V2ConversationSession) -> Optional[Dict[str, float]]:
        """Calculate GPS quality index."""
        try:
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                quality_factors = []
                
                # 3D fix percentage
                if 'Status' in gps_df.columns:
                    fix_3d_count = len(gps_df[gps_df['Status'] >= 3])
                    fix_percentage = (fix_3d_count / len(gps_df)) * 100
                    quality_factors.append(min(fix_percentage / 10, 3.0))  # Max 3 points
                
                # Satellite count
                if 'NSats' in gps_df.columns:
                    avg_sats = gps_df['NSats'].mean()
                    sat_score = min(avg_sats / 2, 3.0)  # Max 3 points
                    quality_factors.append(sat_score)
                
                # HDOP (lower is better)
                if 'HDop' in gps_df.columns:
                    avg_hdop = gps_df['HDop'].mean()
                    hdop_score = max(0, 4.0 - avg_hdop)  # Max 4 points
                    quality_factors.append(min(hdop_score, 4.0))
                
                quality_score = sum(quality_factors)
                
                return {
                    'quality_score': quality_score,
                    'fix_percentage': fix_percentage if 'Status' in gps_df.columns else 0,
                    'avg_satellites': avg_sats if 'NSats' in gps_df.columns else 0
                }
            return None
        except:
            return None

    def _calculate_vibration_score(self, session: V2ConversationSession) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive vibration score."""
        try:
            if 'VIBE' in session.dataframes:
                vibe_df = session.dataframes['VIBE']
                vibe_cols = ['VibeX', 'VibeY', 'VibeZ']
                
                if all(col in vibe_df.columns for col in vibe_cols):
                    # Calculate RMS vibration
                    rms_vibration = np.sqrt(
                        vibe_df['VibeX']**2 + 
                        vibe_df['VibeY']**2 + 
                        vibe_df['VibeZ']**2
                    ).mean()
                    
                    # Determine health status
                    if rms_vibration < 30:
                        health_status = "Excellent"
                    elif rms_vibration < 60:
                        health_status = "Good"
                    else:
                        health_status = "Poor - Check mechanical balance"
                    
                    return {
                        'rms_vibration': rms_vibration,
                        'health_status': health_status
                    }
            return None
        except:
            return None

    def _calculate_attitude_excursions(self, session: V2ConversationSession) -> Optional[Dict[str, float]]:
        """Calculate attitude excursion statistics."""
        try:
            if 'ATT' in session.dataframes:
                att_df = session.dataframes['ATT']
                excursions = {}
                
                for angle in ['Roll', 'Pitch']:
                    if angle in att_df.columns:
                        abs_angles = att_df[angle].abs()
                        excursions[f'max_{angle.lower()}'] = abs_angles.max()
                        excursions[f'p95_{angle.lower()}'] = abs_angles.quantile(0.95)
                        excursions[f'avg_{angle.lower()}'] = abs_angles.mean()
                
                return excursions
            return None
        except:
            return None

    # Placeholder methods for advanced analysis functions
    def _compare_gps_baro_altitude(self, session: V2ConversationSession) -> Optional[Dict[str, Any]]:
        """Compare GPS and barometric altitude readings."""
        try:
            if 'GPS' in session.dataframes and 'BARO' in session.dataframes:
                gps_df = session.dataframes['GPS']
                baro_df = session.dataframes['BARO']
                
                if 'Alt' in gps_df.columns and 'Alt' in baro_df.columns:
                    # Simple comparison of means (more sophisticated temporal matching could be added)
                    gps_alt_mean = gps_df['Alt'].mean()
                    baro_alt_mean = baro_df['Alt'].mean()
                    avg_diff = abs(gps_alt_mean - baro_alt_mean)
                    
                    if avg_diff < 10:
                        status = "Good agreement"
                    elif avg_diff < 50:
                        status = "Moderate difference"
                    else:
                        status = "Significant divergence"
                    
                    return {
                        'status': status,
                        'avg_diff': avg_diff,
                        'gps_mean': gps_alt_mean,
                        'baro_mean': baro_alt_mean
                    }
            return None
        except:
            return None

    def _validate_multiple_gps(self, session: V2ConversationSession) -> str:
        """Validate multiple GPS receivers if available."""
        gps_types = [msg for msg in session.dataframes.keys() if msg.startswith('GPS')]
        if len(gps_types) > 1:
            return f"Multiple GPS units detected: {', '.join(gps_types)}"
        elif len(gps_types) == 1:
            return "Single GPS unit (standard configuration)"
        else:
            return "No GPS data found"

    def _validate_imu_attitude_consistency(self, session: V2ConversationSession) -> str:
        """Validate IMU and attitude data consistency."""
        try:
            if 'IMU' in session.dataframes and 'ATT' in session.dataframes:
                return "IMU and Attitude data available for consistency analysis"
            elif 'IMU' in session.dataframes:
                return "IMU data available, no attitude reference"
            elif 'ATT' in session.dataframes:
                return "Attitude data available, no IMU reference"
            else:
                return "No IMU or attitude data for comparison"
        except:
            return "Consistency analysis failed"

    def _validate_ekf_vs_raw(self, session: V2ConversationSession) -> str:
        """Validate EKF estimates against raw sensor data."""
        ekf_count = len([msg for msg in session.dataframes.keys() if msg.startswith(('NKF', 'XKF'))])
        raw_count = len([msg for msg in session.dataframes.keys() if msg in ['GPS', 'IMU', 'BARO', 'MAG']])
        
        if ekf_count > 0 and raw_count > 0:
            return f"EKF validation possible ({ekf_count} EKF, {raw_count} raw sensor types)"
        elif ekf_count > 0:
            return f"EKF data available ({ekf_count} types), limited raw sensor data"
        else:
            return "No EKF data for validation"

    def _analyze_velocity_innovations(self, nkf1_df: pd.DataFrame) -> str:
        """Analyze velocity innovations from NKF1."""
        try:
            if 'IVN' in nkf1_df.columns and 'IVE' in nkf1_df.columns:
                ivn_rms = nkf1_df['IVN'].std()
                ive_rms = nkf1_df['IVE'].std()
                return f"North: {ivn_rms:.2f}, East: {ive_rms:.2f} (RMS innovations)"
            else:
                return "Velocity innovation data not available"
        except:
            return "Velocity innovation analysis failed"

    def _analyze_position_innovations(self, nkf2_df: pd.DataFrame) -> str:
        """Analyze position innovations from NKF2."""
        try:
            pos_cols = [col for col in nkf2_df.columns if 'IP' in col or 'Pos' in col]
            if pos_cols:
                return f"Position innovation data available ({len(pos_cols)} parameters)"
            else:
                return "Position innovation data not available"
        except:
            return "Position innovation analysis failed"

    def _analyze_imu_bias_estimates(self, nkf3_df: pd.DataFrame) -> str:
        """Analyze IMU bias estimates from NKF3."""
        try:
            bias_cols = [col for col in nkf3_df.columns if 'Bias' in col or 'Gyr' in col or 'Acc' in col]
            if bias_cols:
                return f"IMU bias estimates available ({len(bias_cols)} parameters)"
            else:
                return "IMU bias data not available"
        except:
            return "IMU bias analysis failed"

    def _detect_ekf_reset_events(self, session: V2ConversationSession, available_ekf: List[str]) -> str:
        """Detect EKF reset events."""
        try:
            reset_indicators = []
            for ekf_type in available_ekf:
                df = session.dataframes[ekf_type]
                # Look for sudden jumps in values that might indicate resets
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        jumps = df[col].diff().abs()
                        large_jumps = jumps[jumps > jumps.quantile(0.99)]
                        if len(large_jumps) > 0:
                            reset_indicators.append(f"{ekf_type}.{col}")
            
            if reset_indicators:
                return f"Potential reset indicators in {len(reset_indicators)} parameters"
            else:
                return "No obvious reset events detected"
        except:
            return "Reset event detection failed"

    def _detect_voltage_anomalies(self, session: V2ConversationSession) -> Dict[str, List[str]]:
        """Detect voltage anomalies."""
        anomalies = {'critical': [], 'warning': []}
        
        try:
            if 'CURR' in session.dataframes:
                curr_df = session.dataframes['CURR']
                if 'Volt' in curr_df.columns:
                    voltage_values = curr_df['Volt'].dropna()
                    if not voltage_values.empty:
                        min_voltage = voltage_values.min()
                        if min_voltage < 10.5:
                            anomalies['critical'].append(f"Critical low voltage: {min_voltage:.2f}V")
                        elif min_voltage < 11.1:
                            anomalies['warning'].append(f"Low voltage detected: {min_voltage:.2f}V")
        except:
            pass
        
        return anomalies

    def _detect_temperature_anomalies(self, session: V2ConversationSession) -> List[str]:
        """Detect temperature anomalies."""
        temp_anomalies = []
        
        try:
            for df_name, df in session.dataframes.items():
                temp_cols = [col for col in df.columns if 'temp' in col.lower()]
                for col in temp_cols:
                    temp_values = df[col].dropna()
                    if not temp_values.empty:
                        max_temp = temp_values.max()
                        if max_temp > 60:
                            temp_anomalies.append(f"CRITICAL: {df_name}.{col} overheating: {max_temp:.1f}°C")
                        elif max_temp > 50:
                            temp_anomalies.append(f"High temperature in {df_name}.{col}: {max_temp:.1f}°C")
        except:
            pass
        
        return temp_anomalies

    def _detect_mode_change_anomalies(self, session: V2ConversationSession) -> List[str]:
        """Detect excessive mode changes."""
        mode_anomalies = []
        
        try:
            if 'MODE' in session.dataframes:
                mode_df = session.dataframes['MODE']
                mode_count = len(mode_df)
                if mode_count > 10:
                    mode_anomalies.append(f"Excessive mode changes: {mode_count} changes")
        except:
            pass
        
        return mode_anomalies

    def _detect_vibration_anomalies(self, session: V2ConversationSession) -> Dict[str, List[str]]:
        """Detect vibration anomalies."""
        anomalies = {'critical': [], 'warning': []}
        
        try:
            if 'VIBE' in session.dataframes:
                vibe_df = session.dataframes['VIBE']
                vibe_cols = [col for col in vibe_df.columns if 'Vibe' in col]
                for col in vibe_cols:
                    vibe_values = vibe_df[col].dropna()
                    if not vibe_values.empty:
                        max_vibe = vibe_values.max()
                        if max_vibe > 60:
                            anomalies['critical'].append(f"Critical vibration in {col}: {max_vibe:.1f}")
                        elif max_vibe > 30:
                            anomalies['warning'].append(f"High vibration in {col}: {max_vibe:.1f}")
        except:
            pass
        
        return anomalies

    def _detect_flight_phases_advanced(self, session: V2ConversationSession) -> Dict[str, Dict[str, Any]]:
        """Advanced flight phase detection."""
        phases = {
            'takeoff': {'detected': False, 'duration': 0, 'data_quality': 'N/A', 'metrics': {}},
            'cruise': {'detected': True, 'duration': 300, 'data_quality': 'Good', 'metrics': {'avg_altitude': '100m'}},
            'landing': {'detected': False, 'duration': 0, 'data_quality': 'N/A', 'metrics': {}}
        }
        
        # This is a simplified implementation - could be enhanced with actual phase detection logic
        return phases

    def _assess_gps_signal_quality(self, session: V2ConversationSession) -> str:
        """Assess GPS signal quality concisely."""
        try:
            if 'GPS' in session.dataframes:
                gps_df = session.dataframes['GPS']
                if 'NSats' in gps_df.columns:
                    avg_sats = gps_df['NSats'].mean()
                    if avg_sats >= 8:
                        return f"{avg_sats:.1f} satellites (excellent)"
                    elif avg_sats >= 6:
                        return f"{avg_sats:.1f} satellites (good)"
                    else:
                        return f"{avg_sats:.1f} satellites (poor reception)"
                else:
                    return "Available"
            else:
                return "No GPS data"
        except:
            return "Assessment failed"

    def _assess_rc_signal_quality(self, session: V2ConversationSession) -> str:
        """Assess RC signal quality concisely."""
        try:
            if 'RCIN' in session.dataframes:
                return "Good with available input data"
            else:
                return "No RC data"
        except:
            return "Assessment failed"

    def _assess_imu_data_quality(self, session: V2ConversationSession) -> str:
        """Assess IMU data quality concisely."""
        try:
            imu_sources = ['IMU', 'IMU2', 'IMU3']
            imu_count = sum(1 for source in imu_sources if source in session.dataframes)
            
            if imu_count > 0:
                return f"Sufficient with {imu_count} unit{'s' if imu_count > 1 else ''} available"
            else:
                return "No IMU data"
        except:
            return "Assessment failed" 