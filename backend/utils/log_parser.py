"""
Log parsing utilities for UAV log files.
Handles both optimized and standard parsing methods.
"""

import logging
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from pymavlink import DFReader
from models import V2ConversationSession
from optimized_parser import parse_bin_file_optimized
from config import get_settings

logger = logging.getLogger(__name__)


class LogParserService:
    """Service for parsing UAV log files into DataFrames."""
    
    def __init__(self):
        self.settings = get_settings()

    async def process_log_file(self, session_id: str, file_path: str, session: V2ConversationSession):
        """
        Parses a log file (e.g., .bin) directly into pandas DataFrames using optimized parsing
        and stores them in the session.
        """
        start_time = time.time()
        
        # Check if optimized parsing is enabled
        use_optimized = self.settings.use_optimized_parser
        logger.info(f"[{session_id}] Starting {'optimized' if use_optimized else 'standard'} log file processing for {file_path}")

        try:
            session.dataframes = {}
            session.dataframe_schemas = {}
            session.processing_error = None

            if use_optimized:
                try:
                    # Use the optimized parser
                    logger.info(f"[{session_id}] Using optimized bin file parser")
                    dataframes, schemas = await parse_bin_file_optimized(file_path, session_id)
                    
                    # Store the results
                    session.dataframes = dataframes
                    session.dataframe_schemas = schemas
                    
                    processing_time = time.time() - start_time
                    total_messages = sum(len(df) for df in dataframes.values())
                    
                    logger.info(f"[{session_id}] OPTIMIZED PARSING COMPLETE!")
                    logger.info(f"[{session_id}] Processing time: {processing_time:.2f}s")
                    logger.info(f"[{session_id}] Total messages: {total_messages:,}")
                    logger.info(f"[{session_id}] Message types: {len(dataframes)}")
                    logger.info(f"[{session_id}] Throughput: {total_messages/processing_time:.0f} messages/second")
                    logger.info(f"[{session_id}] DataFrames created for: {list(dataframes.keys())}")
                    
                except Exception as e:
                    logger.warning(f"[{session_id}] Optimized parsing failed: {e}, falling back to standard parsing")
                    # Fall through to standard parsing
                    use_optimized = False
            
            if not use_optimized:
                # Use standard parsing method
                await self._standard_parse_log_file(session_id, file_path, session)
                
                processing_time = time.time() - start_time
                total_messages = sum(len(df) for df in session.dataframes.values())
                
                logger.info(f"[{session_id}] Standard parsing completed in {processing_time:.2f}s")
                logger.info(f"[{session_id}] Total messages: {total_messages:,}, Message types: {len(session.dataframes)}")

        except Exception as e:
            session.processing_error = str(e)
            logger.error(f"[{session_id}] Error in log file processing {file_path}: {e}", exc_info=True)
            
        finally:
            session.is_processing = False
            session.last_updated = datetime.now()

    async def _standard_parse_log_file(self, session_id: str, file_path: str, session: V2ConversationSession):
        """Standard parsing method using the original DFReader approach."""
        logger.info(f"[{session_id}] Using standard DFReader parsing method")
        
        log = DFReader.DFReader_binary(file_path)
        
        data = {}
        message_count = 0
        while True:
            msg = log.recv_msg()
            if msg is None:
                break
            msg_type = msg.get_type()
            if msg_type not in data:
                data[msg_type] = []
            data[msg_type].append(msg.to_dict())
            message_count += 1
            
            # Log progress every 2000 messages for less noise
            if message_count % 2000 == 0:
                logger.info(f"[{session_id}] Standard parsing: processed {message_count} messages...")

        logger.info(f"[{session_id}] Standard parsing finished reading {message_count} messages, found {len(data)} message types")

        for msg_type, msg_list in data.items():
            if not msg_list:
                continue
            
            df = pd.DataFrame(msg_list)
            
            if 'TimeUS' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TimeUS'], unit='us')
            elif 'time_boot_ms' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time_boot_ms'], unit='ms')

            session.dataframes[msg_type] = df
            
            session.dataframe_schemas[msg_type] = {
                "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "description": f"Contains {msg_type} data from the flight log."
            }
        
        logger.info(f"[{session_id}] Standard parsing completed successfully")


def get_data_summary(session: V2ConversationSession) -> Dict[str, Any]:
    """Generate a comprehensive summary of the available data for better context."""
    # Ensure we always return a valid dictionary even if session or dataframes are None/empty
    if not session or not hasattr(session, 'dataframes') or not session.dataframes:
        return {
            "message_types": 0,
            "total_records": 0,
            "time_range": {
                "start": "Unknown",
                "end": "Unknown", 
                "duration_minutes": 0
            },
            "key_metrics": {}
        }
    
    summary = {
        "message_types": len(session.dataframes),
        "total_records": sum(len(df) for df in session.dataframes.values()),
        "time_range": {
            "start": "Unknown",
            "end": "Unknown",
            "duration_minutes": 0
        },
        "key_metrics": {}
    }
    
    try:
        # Find time range
        timestamps = []
        for df in session.dataframes.values():
            if 'timestamp' in df.columns:
                timestamps.extend(df['timestamp'].dropna().tolist())
        
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            summary["time_range"] = {
                "start": min_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end": max_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_minutes": (max_time - min_time).total_seconds() / 60
            }
        
        # Key metrics
        if 'GPS' in session.dataframes:
            gps_df = session.dataframes['GPS']
            if 'Alt' in gps_df.columns:
                summary["key_metrics"]["max_altitude"] = float(gps_df['Alt'].max())
            if 'Spd' in gps_df.columns:
                summary["key_metrics"]["max_speed"] = float(gps_df['Spd'].max())
        
        if 'BARO' in session.dataframes:
            baro_df = session.dataframes['BARO']
            if 'Alt' in baro_df.columns:
                summary["key_metrics"]["max_baro_altitude"] = float(baro_df['Alt'].max())
        
    except Exception as e:
        logger.debug(f"Error generating data summary: {e}")
        # Ensure we still have a valid time_range structure
        if "time_range" not in summary or summary["time_range"] is None:
            summary["time_range"] = {
                "start": "Unknown",
                "end": "Unknown",
                "duration_minutes": 0
            }
    
    return summary 