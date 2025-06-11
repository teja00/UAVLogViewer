#!/usr/bin/env python3
"""
Performance testing script for bin file parsing optimization.
Compares original vs optimized parsing methods.
"""

import asyncio
import time
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from pymavlink import DFReader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our optimized parser
try:
    from optimized_parser import parse_bin_file_optimized
    OPTIMIZED_AVAILABLE = True
except ImportError:
    logger.warning("Optimized parser not available, will only test original method")
    OPTIMIZED_AVAILABLE = False

class PerformanceTestResult:
    """Container for test results"""
    def __init__(self, method: str, file_path: str, file_size_mb: float):
        self.method = method
        self.file_path = file_path
        self.file_size_mb = file_size_mb
        self.parse_time_seconds = 0
        self.total_messages = 0
        self.message_types = 0
        self.memory_usage_mb = 0
        self.success = False
        self.error = None
        
    @property
    def throughput_msg_per_sec(self) -> float:
        if self.parse_time_seconds > 0:
            return self.total_messages / self.parse_time_seconds
        return 0
    
    @property
    def throughput_mb_per_sec(self) -> float:
        if self.parse_time_seconds > 0:
            return self.file_size_mb / self.parse_time_seconds
        return 0
    
    def __str__(self) -> str:
        if not self.success:
            return f"{self.method} - FAILED: {self.error}"
        
        return (f"{self.method}:\n"
                f"  ⏱️  Parse Time: {self.parse_time_seconds:.2f}s\n"
                f"  📊 Messages: {self.total_messages:,} ({self.message_types} types)\n"
                f"  🚀 Throughput: {self.throughput_msg_per_sec:.0f} msg/s, {self.throughput_mb_per_sec:.1f} MB/s\n"
                f"  💾 Memory: ~{self.memory_usage_mb:.1f} MB")

async def parse_with_original_method(file_path: str) -> PerformanceTestResult:
    """Test the original parsing method"""
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    result = PerformanceTestResult("Original DFReader", file_path, file_size_mb)
    
    try:
        start_time = time.time()
        
        # Original parsing logic from chat_service_v2.py
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

        # Convert to DataFrames (like in original)
        dataframes = {}
        for msg_type, msg_list in data.items():
            if not msg_list:
                continue
            
            df = pd.DataFrame(msg_list)
            
            if 'TimeUS' in df.columns:
                df['timestamp'] = pd.to_datetime(df['TimeUS'], unit='us')
            elif 'time_boot_ms' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time_boot_ms'], unit='ms')

            dataframes[msg_type] = df
        
        result.parse_time_seconds = time.time() - start_time
        result.total_messages = message_count
        result.message_types = len(dataframes)
        result.success = True
        
        # Rough memory estimation (DataFrames in memory)
        result.memory_usage_mb = sum(df.memory_usage(deep=True).sum() for df in dataframes.values()) / (1024 * 1024)
        
    except Exception as e:
        result.error = str(e)
        result.success = False
        logger.error(f"Original parsing failed: {e}")
    
    return result

async def parse_with_optimized_method(file_path: str) -> PerformanceTestResult:
    """Test the optimized parsing method"""
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    result = PerformanceTestResult("Optimized Parser", file_path, file_size_mb)
    
    if not OPTIMIZED_AVAILABLE:
        result.error = "Optimized parser not available"
        result.success = False
        return result
    
    try:
        start_time = time.time()
        
        # Use optimized parser
        dataframes, schemas = await parse_bin_file_optimized(file_path, "test-session")
        
        result.parse_time_seconds = time.time() - start_time
        result.total_messages = sum(len(df) for df in dataframes.values())
        result.message_types = len(dataframes)
        result.success = True
        
        # Memory estimation
        result.memory_usage_mb = sum(df.memory_usage(deep=True).sum() for df in dataframes.values()) / (1024 * 1024)
        
    except Exception as e:
        result.error = str(e)
        result.success = False
        logger.error(f"Optimized parsing failed: {e}")
    
    return result

async def run_performance_test(file_path: str) -> Dict[str, PerformanceTestResult]:
    """Run performance comparison test"""
    logger.info(f"🧪 Starting performance test on: {file_path}")
    
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    logger.info(f"📁 File size: {file_size_mb:.1f} MB")
    
    results = {}
    
    # Test original method
    logger.info("🔄 Testing original parsing method...")
    results['original'] = await parse_with_original_method(file_path)
    
    # Test optimized method
    if OPTIMIZED_AVAILABLE:
        logger.info("⚡ Testing optimized parsing method...")
        results['optimized'] = await parse_with_optimized_method(file_path)
    
    return results

def print_comparison_report(results: Dict[str, PerformanceTestResult]):
    """Print a detailed comparison report"""
    print("\n" + "="*80)
    print("🏁 PERFORMANCE TEST RESULTS")
    print("="*80)
    
    for method_name, result in results.items():
        print(f"\n{result}")
    
    # Calculate improvements if both methods succeeded
    if len(results) == 2 and all(r.success for r in results.values()):
        original = results['original']
        optimized = results['optimized']
        
        time_improvement = ((original.parse_time_seconds - optimized.parse_time_seconds) / original.parse_time_seconds) * 100
        throughput_improvement = ((optimized.throughput_msg_per_sec - original.throughput_msg_per_sec) / original.throughput_msg_per_sec) * 100
        
        print(f"\n🎯 IMPROVEMENT ANALYSIS:")
        print(f"⏱️  Time reduction: {time_improvement:.1f}% faster")
        print(f"🚀 Throughput gain: {throughput_improvement:.1f}% more messages/sec")
        
        if time_improvement > 0:
            print(f"✅ Optimized parser is {time_improvement:.1f}% faster!")
        else:
            print(f"⚠️  Optimized parser is {abs(time_improvement):.1f}% slower")
    
    print("\n" + "="*80)

async def main():
    """Main testing function"""
    import sys
    
    # Check if file path provided
    if len(sys.argv) < 2:
        print("Usage: python performance_test.py <path_to_bin_file>")
        print("\nExample:")
        print("  python performance_test.py ~/Downloads/flight_log.bin")
        
        # Look for test files in common locations
        common_paths = [
            Path.home() / "Downloads",
            Path("test/testlogfiles"),  
            Path("../test/testlogfiles"),
            Path(".")
        ]
        
        print(f"\n🔍 Looking for .bin files in common locations...")
        found_files = []
        for path in common_paths:
            if path.exists():
                bin_files = list(path.glob("*.bin"))
                found_files.extend(bin_files)
        
        if found_files:
            print("Found these .bin files:")
            for i, f in enumerate(found_files[:5], 1):  # Show max 5 files
                print(f"  {i}. {f}")
            
            if len(found_files) > 5:
                print(f"  ... and {len(found_files) - 5} more")
                
            print(f"\nTo test with one of these files, run:")
            print(f"  python performance_test.py '{found_files[0]}'")
        else:
            print("No .bin files found in common locations.")
        
        return
    
    file_path = sys.argv[1]
    
    # Validate file exists
    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        return
    
    if not file_path.lower().endswith('.bin'):
        print(f"⚠️  Warning: File doesn't have .bin extension: {file_path}")
        print("Continuing anyway...")
    
    try:
        # Run the performance test
        results = await run_performance_test(file_path)
        
        # Print results
        print_comparison_report(results)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 