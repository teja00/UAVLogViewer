import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np
import pandas as pd
from pymavlink import DFReader, mavutil

logger = logging.getLogger(__name__)

class OptimizedBinParser:
    """
    High-performance bin file parser with multiple optimization strategies.
    Maintains compatibility with existing chat_service_v2.py interface.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parser with configurable thread pool.
        
        Args:
            max_workers: Number of worker threads. Defaults to CPU count.
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
    def __del__(self):
        """Cleanup executor on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    async def parse_bin_file_async(self, file_path: str, session_id: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Asynchronously parse bin file with multiple optimization strategies.
        
        Args:
            file_path: Path to the bin file
            session_id: Session ID for logging
            
        Returns:
            Tuple of (dataframes_dict, schemas_dict)
        """
        start_time = time.time()
        
        if session_id:
            logger.info(f"[{session_id}] Starting optimized parsing for {file_path}")
        
        # Strategy 1: Try memory-mapped parsing first (fastest for smaller files)
        try:
            dataframes, schemas = await self._parse_with_memory_mapping(file_path, session_id)
            if session_id:
                logger.info(f"[{session_id}] Memory-mapped parsing completed in {time.time() - start_time:.2f}s")
            return dataframes, schemas
        except Exception as e:
            if session_id:
                logger.warning(f"[{session_id}] Memory-mapped parsing failed: {e}, falling back to chunked parsing")
        
        # Strategy 2: Chunked parallel parsing (better for larger files)
        try:
            dataframes, schemas = await self._parse_with_chunking(file_path, session_id)
            if session_id:
                logger.info(f"[{session_id}] Chunked parsing completed in {time.time() - start_time:.2f}s")
            return dataframes, schemas
        except Exception as e:
            if session_id:
                logger.warning(f"[{session_id}] Chunked parsing failed: {e}, falling back to optimized sequential")
        
        # Strategy 3: Optimized sequential parsing (fallback)
        dataframes, schemas = await self._parse_optimized_sequential(file_path, session_id)
        if session_id:
            logger.info(f"[{session_id}] Optimized sequential parsing completed in {time.time() - start_time:.2f}s")
        return dataframes, schemas
    
    async def _parse_with_memory_mapping(self, file_path: str, session_id: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Ultra-fast parsing using memory mapping for smaller files.
        """
        def _parse_mmap():
            # Pre-allocate data structures
            data = {}
            message_count = 0
            
            # Use mavutil for better performance with small files
            mlog = mavutil.mavlink_connection(file_path, dialect="ardupilotmega")
            
            # Batch message reading
            batch_size = 1000
            while True:
                batch = []
                for _ in range(batch_size):
                    msg = mlog.recv_match(blocking=False)
                    if msg is None:
                        break
                    batch.append(msg)
                
                if not batch:
                    break
                
                # Process batch
                for msg in batch:
                    msg_type = msg.get_type()
                    if msg_type not in data:
                        data[msg_type] = []
                    data[msg_type].append(msg.to_dict())
                    message_count += 1
                
                if session_id and message_count % 5000 == 0:
                    logger.info(f"[{session_id}] Memory-mapped: processed {message_count} messages...")
            
            return data, message_count
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        data, message_count = await loop.run_in_executor(self.executor, _parse_mmap)
        
        # Convert to DataFrames with optimized dtypes
        dataframes = {}
        schemas = {}
        
        for msg_type, msg_list in data.items():
            if not msg_list:
                continue
                
            # Create DataFrame with optimized dtypes
            df = self._create_optimized_dataframe(msg_list, msg_type)
            
            # Add timestamp column
            df = self._add_timestamp_column(df)
            
            dataframes[msg_type] = df
            schemas[msg_type] = {
                "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "description": f"Contains {msg_type} data from the flight log."
            }
        
        return dataframes, schemas
    
    async def _parse_with_chunking(self, file_path: str, session_id: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Parallel chunked parsing for larger files.
        Note: This is a simplified implementation. For production use with very large files,
        proper message boundary detection would be needed.
        """
        def _parse_chunk_safe(file_path: str, chunk_id: int) -> Dict[str, List[Dict]]:
            """
            Parse a logical chunk of messages using a separate connection.
            This avoids file boundary issues by using separate DFReader instances.
            """
            data = {}
            
            try:
                # Each worker gets its own DFReader instance
                log = DFReader.DFReader_binary(file_path)
                
                # Skip messages based on chunk_id to distribute work
                # This is a simplified approach - could be more sophisticated
                skip_count = chunk_id * 5000  # Skip 5000 messages per chunk
                read_count = 5000  # Read up to 5000 messages per chunk
                
                message_count = 0
                processed_count = 0
                
                while processed_count < read_count:
                    msg = log.recv_msg()
                    if msg is None:
                        break
                    
                    # Skip messages for load balancing
                    if message_count < skip_count:
                        message_count += 1
                        continue
                    
                    msg_type = msg.get_type()
                    if msg_type not in data:
                        data[msg_type] = []
                    data[msg_type].append(msg.to_dict())
                    
                    processed_count += 1
                    message_count += 1
                        
            except Exception as e:
                logger.warning(f"Chunk {chunk_id} parsing error: {e}")
                
            return data
        
        # Create chunk tasks based on logical message chunks rather than file size
        num_chunks = min(self.max_workers, 4)  # Don't create too many chunks
        
        tasks = []
        loop = asyncio.get_event_loop()
        
        for chunk_id in range(num_chunks):
            task = loop.run_in_executor(
                self.executor, 
                _parse_chunk_safe, 
                file_path, 
                chunk_id
            )
            tasks.append(task)
        
        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*tasks)
        
        # Merge results from all chunks
        merged_data = {}
        total_messages = 0
        
        for chunk_data in chunk_results:
            for msg_type, msg_list in chunk_data.items():
                if msg_type not in merged_data:
                    merged_data[msg_type] = []
                merged_data[msg_type].extend(msg_list)
                total_messages += len(msg_list)
        
        if session_id:
            logger.info(f"[{session_id}] Chunked parsing processed {total_messages} messages from {len(chunk_results)} chunks")
        
        # Convert to DataFrames
        dataframes = {}
        schemas = {}
        
        for msg_type, msg_list in merged_data.items():
            if not msg_list:
                continue
                
            df = self._create_optimized_dataframe(msg_list, msg_type)
            df = self._add_timestamp_column(df)
            
            dataframes[msg_type] = df
            schemas[msg_type] = {
                "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "description": f"Contains {msg_type} data from the flight log."
            }
        
        return dataframes, schemas
    
    async def _parse_optimized_sequential(self, file_path: str, session_id: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Optimized sequential parsing with batching and efficient memory usage.
        """
        def _parse_sequential():
            log = DFReader.DFReader_binary(file_path)
            
            # Use more efficient data structures
            data = {}
            message_count = 0
            batch_size = 2000  # Larger batch size for better performance
            
            while True:
                # Read messages in batches
                batch = []
                for _ in range(batch_size):
                    msg = log.recv_msg()
                    if msg is None:
                        break
                    batch.append(msg)
                
                if not batch:
                    break
                
                # Process batch efficiently
                for msg in batch:
                    msg_type = msg.get_type()
                    if msg_type not in data:
                        data[msg_type] = []
                    data[msg_type].append(msg.to_dict())
                    message_count += 1
                
                # Progress logging
                if session_id and message_count % 10000 == 0:
                    logger.info(f"[{session_id}] Optimized sequential: processed {message_count} messages...")
            
            return data, message_count
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        data, message_count = await loop.run_in_executor(self.executor, _parse_sequential)
        
        # Convert to DataFrames with optimizations
        dataframes = {}
        schemas = {}
        
        for msg_type, msg_list in data.items():
            if not msg_list:
                continue
                
            df = self._create_optimized_dataframe(msg_list, msg_type)
            df = self._add_timestamp_column(df)
            
            dataframes[msg_type] = df
            schemas[msg_type] = {
                "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "description": f"Contains {msg_type} data from the flight log."
            }
        
        return dataframes, schemas
    
    def _create_optimized_dataframe(self, msg_list: List[Dict], msg_type: str) -> pd.DataFrame:
        """
        Create DataFrame with optimized dtypes to reduce memory usage and improve performance.
        """
        if not msg_list:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(msg_list)
        
        # Optimize dtypes based on message type and content
        for col in df.columns:
            if col in ['TimeUS', 'time_boot_ms', 'time_usec']:
                # Keep timestamp columns as int64 for precision
                continue
            elif df[col].dtype == 'object':
                # Try to convert object columns to more efficient types
                try:
                    # Try numeric conversion first
                    numeric_df = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_df.isna().all():
                        # Check if can be integer
                        if numeric_df.notna().all() and (numeric_df % 1 == 0).all():
                            df[col] = numeric_df.astype('int32')
                        else:
                            df[col] = numeric_df.astype('float32')
                    else:
                        # Keep as categorical for repeated strings
                        if df[col].nunique() < len(df[col]) * 0.5:
                            df[col] = df[col].astype('category')
                except:
                    pass
            elif df[col].dtype == 'float64':
                # Downcast to float32 if possible
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                # Downcast to smaller int if possible
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def _add_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standardized timestamp column."""
        if 'TimeUS' in df.columns:
            df['timestamp'] = pd.to_datetime(df['TimeUS'], unit='us')
        elif 'time_boot_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time_boot_ms'], unit='ms')
        elif 'time_usec' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time_usec'], unit='us')
        
        return df

# Global parser instance
_parser_instance = None

def get_parser() -> OptimizedBinParser:
    """Get singleton parser instance"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = OptimizedBinParser()
    return _parser_instance

async def parse_bin_file_optimized(file_path: str, session_id: str = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Convenience function for optimized bin file parsing.
    
    Args:
        file_path: Path to the bin file
        session_id: Session ID for logging
        
    Returns:
        Tuple of (dataframes_dict, schemas_dict)
    """
    parser = get_parser()
    return await parser.parse_bin_file_async(file_path, session_id) 