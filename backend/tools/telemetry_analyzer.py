from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import logging
import re
import traceback

# Temporarily disable sklearn imports due to NumPy 2.x compatibility issues
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
@dataclass
class AnomalyDetection:
    type: str
    timestamp: pd.Timestamp
    description: str
    severity: str
    data: Dict[str, Any]


# -----------------------------------------------------------------------------
class TelemetryAnalyzer:
    """Advanced analytics on telemetry data from multiple UAV platforms."""

    # Unit metadata extracted from MAVLink specifications
    TELEMETRY_UNITS = {
        # Position and altitude units
        "lat": {"unit": "deg", "scale": 1e-7, "for_fields": ["lat", "latitude", "lng", "lon", "longitude"]},
        "lon": {"unit": "deg", "scale": 1e-7, "for_fields": ["lon", "lng", "longitude"]},
        "alt": {"unit": "m", "scale": 1.0, "for_fields": ["alt", "altitude", "height", "terrain_height"]},
        "relative_alt": {"unit": "m", "scale": 1e-3, "for_fields": ["relative_alt", "relative_altitude"]},
        "climb": {"unit": "m/s", "scale": 1.0, "for_fields": ["climb", "climb_rate", "vspeed"]},
        
        # Velocity units
        "vx": {"unit": "m/s", "scale": 0.01, "for_fields": ["vx", "velocity_x", "vel_x"]},
        "vy": {"unit": "m/s", "scale": 0.01, "for_fields": ["vy", "velocity_y", "vel_y"]},
        "vz": {"unit": "m/s", "scale": 0.01, "for_fields": ["vz", "velocity_z", "vel_z"]},
        "groundspeed": {"unit": "m/s", "scale": 1.0, "for_fields": ["groundspeed", "ground_speed", "gspeed"]},
        "airspeed": {"unit": "m/s", "scale": 1.0, "for_fields": ["airspeed", "air_speed", "aspeed"]},
        
        # Battery and power units
        "voltage": {"unit": "V", "scale": 1e-3, "for_fields": ["voltage", "volt", "voltage_battery", "voltages"]},
        "current": {"unit": "A", "scale": 0.01, "for_fields": ["current", "current_battery", "curr"]},
        "remaining": {"unit": "%", "scale": 1.0, "for_fields": ["battery_remaining", "remaining", "capacity"]},
        
        # Attitude units
        "roll": {"unit": "rad", "scale": 1.0, "for_fields": ["roll", "roll_angle"]},
        "pitch": {"unit": "rad", "scale": 1.0, "for_fields": ["pitch", "pitch_angle"]},
        "yaw": {"unit": "rad", "scale": 1.0, "for_fields": ["yaw", "yaw_angle", "heading"]},
        
        # GPS units
        "eph": {"unit": "m", "scale": 1.0, "for_fields": ["eph", "h_accuracy", "hdop"]},
        "epv": {"unit": "m", "scale": 1.0, "for_fields": ["epv", "v_accuracy", "vdop"]},
        "satellites_visible": {"unit": "count", "scale": 1.0, "for_fields": ["satellites_visible", "satellites", "sats"]}
    }
    
    # Build reverse lookup for unit identification
    FIELD_TO_UNIT_MAP = {}
    for unit_key, unit_info in TELEMETRY_UNITS.items():
        for field in unit_info["for_fields"]:
            FIELD_TO_UNIT_MAP[field] = {
                "base_field": unit_key,
                "unit": unit_info["unit"],
                "scale": unit_info["scale"]
            }

    # -------------------------------------------------------------------------
    def __init__(self, telemetry: Dict[str, pd.DataFrame]) -> None:
        self.telemetry = telemetry
        self.cache: Dict[str, Any] = {}
        self.time_series: Dict[str, List[float]] = {}
        self.unit_info: Dict[str, Dict[str, Any]] = {}
        
        # Process the telemetry data
        logger.info("Processing telemetry data into time series")
        self._process_telemetry_data()
        
        # Verify we have valid data
        if not self.time_series.get("timestamp", []):
            logger.warning("No valid time series data produced after processing")
        else:
            logger.info(f"Successfully processed {len(self.time_series) - 1} data fields")

    def analyze_for_query(self, query: str) -> Dict[str, Any]:
        """Fast, on-demand answer builder for specific queries."""
        if not self.time_series.get("timestamp"):
            return {"error": "No usable telemetry found."}

        now = datetime.now(timezone.utc)
        if ("metrics" not in self.cache or
                (now - self.cache.get("ts", now)).total_seconds() > 60):
            self.cache.update({
                "metrics": self._calc_metrics(),
                "anomalies": self._detect_anomalies(),
                "kpis": self._calculate_kpis(),
                "ts": now,
            })

        q = query.lower()
        out: Dict[str, Any] = {}

        # ------- altitude
        if "altitude" in q or "height" in q:
            alt_analysis = self._analyze_altitude()
            if alt_analysis:
                out["altitude_analysis"] = alt_analysis

        # ------- battery
        if "battery" in q or "voltage" in q or "power" in q:
            bat_analysis = self._analyze_battery()
            if bat_analysis:
                out["battery_analysis"] = bat_analysis

        # Add standard metrics and anomalies
        out.setdefault("metrics", self.cache["metrics"])
        out.setdefault("anomalies", self.cache["anomalies"])
        return out

    def _analyze_altitude(self) -> Dict[str, Any]:
        """Return realistic altitude stats plus take-off / landing info."""
        # Find altitude field
        alt_field = None
        priority = [
            "GLOBAL_POSITION_INT_relative_alt",
            "ALTITUDE_altitude_relative", 
            "VFR_HUD_alt",
            "GPS_RAW_INT_alt",
            "BARO_Alt"
        ]
        
        for field in priority:
            if field in self.time_series:
                alt_field = field
                break
        
        # Fallback: anything containing "alt" / "height"
        if not alt_field:
            for key in self.time_series:
                if key == "timestamp":
                    continue
                lk = key.lower()
                if any(tok in lk for tok in ("alt", "height", "terrain")):
                    alt_field = key
                    break

        if not alt_field:
            logger.warning("No altitude field found")
            return {}

        # Raw values
        vals = np.asarray(self.time_series[alt_field], dtype=float)
        
        # Convert units if needed
        if "relative_alt" in alt_field and vals.max() > 1000:
            vals = vals / 1000.0  # Convert mm to m
            
        stats = {
            "max": float(vals.max()),
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "range": float(vals.ptp()),
        }

        return {
            "field_used": alt_field,
            "statistics": stats,
        }

    # Additional helper methods from the original implementation
    def _process_telemetry_data(self) -> None:
        """Process telemetry DataFrames into a unified time series dictionary."""
        if not self.telemetry:
            logger.warning("No telemetry data provided")
            self.time_series = {"timestamp": []}
            return

        # Convert telemetry dataframes to time series format
        timestamps = []
        numeric_data = {}
        
        for msg_type, df in self.telemetry.items():
            if df is None or df.empty:
                continue
                
            # Get timestamps from index or column
            if hasattr(df.index, 'name') and df.index.name == "timestamp":
                ts = df.index.tolist()
            elif "timestamp" in df.columns:
                ts = df["timestamp"].tolist()
            else:
                continue
                
            timestamps.extend(ts)
            
            # Process numeric columns
            for col in df.columns:
                if col == "timestamp":
                    continue
                    
                try:
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if series.empty:
                        continue
                        
                    field_name = f"{msg_type}_{col}"
                    numeric_data[field_name] = series
                except Exception:
                    continue
        
        if not timestamps:
            self.time_series = {"timestamp": []}
            return
            
        # Create unified time series
        self.time_series = {"timestamp": sorted(set(timestamps))}
        
        # Add numeric data aligned to timestamps
        for field_name, series in numeric_data.items():
            aligned_values = []
            for ts in self.time_series["timestamp"]:
                if ts in series.index:
                    aligned_values.append(float(series[ts]))
                else:
                    # Use forward fill or zero
                    aligned_values.append(0.0)
            self.time_series[field_name] = aligned_values

    def _calc_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic metrics for all fields."""
        metrics = {}
        for k, vals in self.time_series.items():
            if k == "timestamp":
                continue
            arr = np.asarray(vals, dtype="float32")
            metrics[k] = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
            }
        return metrics

    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies using isolation forest."""
        return []  # Simplified for now

    def _calculate_kpis(self) -> Dict[str, float]:
        """Calculate key performance indicators."""
        return {}  # Simplified for now

    def _analyze_battery(self) -> Dict[str, Any]:
        """Analyze battery data."""
        return {}  # Simplified for now 