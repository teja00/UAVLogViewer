"""
Documentation utilities for ArduPilot message types.
Handles dynamic fetching and caching of documentation.
"""

import logging
import time
import re
import httpx
from bs4 import BeautifulSoup
from typing import Dict

logger = logging.getLogger(__name__)


class DocumentationService:
    """Service for fetching and caching ArduPilot documentation."""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=10)
        
        # Static fallback ArduPilot message documentation
        self.ardupilot_messages = {
            "ACC": "IMU accelerometer data - contains AccX, AccY, AccZ acceleration values",
            "ADSB": "Automatic Dependent Surveillance-Broadcast detected vehicle information",
            "AETR": "Normalised pre-mixer control surface outputs - Aileron, Elevator, Throttle, Rudder",
            "AHR2": "Backup AHRS data - Roll, Pitch, Yaw, Alt, Lat, Lng and quaternion components",
            "ATT": "Attitude data - Roll, Pitch, Yaw from the attitude controller",
            "BARO": "Barometer data - Alt (altitude), Press (pressure), Temp (temperature)",
            "CURR": "Battery/Power data - Volt (voltage), Curr (current), CurrTot (total current)",
            "GPS": "GPS position data - Lat, Lng, Alt, Spd (speed), GCrs (ground course)",
            "GPS2": "Secondary GPS data when multiple GPS units are present",
            "IMU": "Inertial measurement unit data - AccX, AccY, AccZ, GyrX, GyrY, GyrZ",
            "MODE": "Flight mode changes - Mode number and asText description",
            "MSG": "Text messages and alerts from the autopilot system",
            "PARM": "Parameter values set in the autopilot",
            "POS": "Position estimates from the EKF - Lat, Lng, Alt",
            "RCIN": "RC input values from transmitter channels",
            "RCOU": "RC output values to servos and motors",
            "VIBE": "Vibration levels affecting IMU performance",
            "XKF1": "Extended Kalman Filter states and innovations",
            "XKF2": "More EKF data including wind estimates",
            "XKF3": "EKF innovation variances and health monitoring",
            "XKF4": "EKF timing and processing information"
        }
        
        # Cache for dynamic documentation
        self.documentation_cache = {}
        self.last_doc_fetch = {}

    async def _check_url_alive(self, url: str) -> bool:
        """Check if a URL is alive and reachable with a quick HEAD request."""
        try:
            response = await self.http_client.head(url, timeout=5)
            return response.status_code < 400
        except Exception:
            return False

    async def fetch_ardupilot_documentation(self, message_type: str) -> str:
        """
        Dynamically fetch ArduPilot documentation for specific message types.
        Returns cached static documentation as fallback.
        """
        try:
            # Try to fetch from ArduPilot docs website
            url = "https://ardupilot.org/plane/docs/logmessages.html"
            
            # Check cache first (cache for 1 hour) 
            cache_key = f"{message_type}_{url}"
            if (cache_key in self.documentation_cache and 
                cache_key in self.last_doc_fetch and 
                time.time() - self.last_doc_fetch[cache_key] < 3600):
                return self.documentation_cache[cache_key]
            
            # Check if URL is alive before making the full request
            if not await self._check_url_alive(url):
                logger.debug(f"ArduPilot documentation URL is not reachable, skipping fetch for {message_type}")
                # Fallback to static documentation immediately
                return self.ardupilot_messages.get(message_type, f"Data related to {message_type} from the flight log.")
            
            response = await self.http_client.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for message type documentation
                text = soup.get_text()
                pattern = rf'{message_type}[:\s]+(.*?)(?=\n[A-Z][A-Z][A-Z0-9]*[:\s]|$)'
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                
                if match:
                    doc = match.group(1).strip()[:500]  # Limit length
                    self.documentation_cache[cache_key] = doc
                    self.last_doc_fetch[cache_key] = time.time()
                    return doc
            
        except Exception as e:
            logger.debug(f"Failed to fetch dynamic documentation for {message_type}: {e}")
        
        # Fallback to static documentation
        return self.ardupilot_messages.get(message_type, f"Data related to {message_type} from the flight log.") 