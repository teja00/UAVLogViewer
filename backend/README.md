# UAV Log Analyzer - V2 Agentic Chat Backend

An OpenAI-powered agentic chatbot backend for analyzing UAV flight telemetry data. This V2 backend processes binary log files directly and provides intelligent analysis using GPT models with comprehensive ArduPilot knowledge and advanced tool-calling capabilities.

## Architecture

The V2 backend provides enhanced agentic analysis capabilities:
- **Direct log file processing** (.bin, .tlog files)
- **High-performance parsing** with optimized DataFrame creation
- **Agentic AI analysis** using OpenAI's tool-calling features
- **Advanced flight data tools** for comprehensive analysis
- **Session-based conversations** with persistent data

## Key Features

- **Enhanced Log Processing**: Direct parsing of ArduPilot .bin/.tlog files with optimized performance
- **Agentic Analysis**: Uses OpenAI's tool-calling for dynamic analysis with multiple specialized tools
- **ArduPilot Expertise**: Built-in understanding of 25+ ArduPilot message types
- **Advanced Tools**: Anomaly detection, flight phase analysis, event detection, timeline analysis
- **High Performance**: Optimized parsing with memory mapping and parallel processing
- **Session Management**: Maintains conversation context and flight data across requests
- **Real-time Analysis**: Fast API responses for interactive chat experience

## Quick Start

### 1. Environment Setup

```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
# Start the server directly
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Upload and Analyze Log Files

```bash
# Upload a log file
curl -X POST -F 'file=@your_flight_log.bin' http://localhost:8000/sessions/upload-log

# This returns a session_id, then use it to chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze this flight and identify any issues",
    "session_id": "your-session-id-here"
  }'
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check and configuration status
- `POST /sessions/upload-log` - Upload .bin/.tlog files for processing
- `POST /chat` - Agentic chat endpoint with advanced analysis tools
- `GET /sessions/{session_id}` - Get session information and processing status
- `DELETE /sessions/{session_id}` - Clear session conversation history

## Data Flow

```
Log File Upload → Background Processing → Agentic Analysis → Intelligent Response
┌─────────────────┐        ┌──────────────────────────┐
│ 1. Upload .bin  │        │                          │
│    file via API │        │                          │
└─────────────────┘        │                          │
         │                 │                          │
┌─────────────────┐        │ ┌──────────────────────┐ │
│ 2. Optimized    │───────▶│ │ High-Performance     │ │
│    parsing to   │        │ │ Parsing:             │ │
│    DataFrames   │        │ │ - Memory mapping     │ │
└─────────────────┘        │ │ - Parallel chunks    │ │
         │                 │ │ - Optimized dtypes   │ │
┌─────────────────┐        │ └──────────────────────┘ │
│ 4. Interactive │◀───────│ ┌──────────────────────┐ │
│    chat with AI │        │ │ 3. Agentic Analysis  │ │
│    analysis     │        │ │ - Tool calling       │ │
└─────────────────┘        │ │ - Event detection    │ │
                           │ │ - Anomaly analysis   │ │
                           │ │ - Flight insights    │ │
                           │ └──────────────────────┘ │
                           └──────────────────────────┘
```

## Example Usage

### 1. Upload a Log File

```bash
curl -X POST -F 'file=@flight_2024_01_15.bin' http://localhost:8000/sessions/upload-log
# Returns: {"session_id": "abc123", "message": "File upload successful, processing has started."}
```

### 2. Wait for Processing (or check status)

```bash
curl http://localhost:8000/sessions/abc123
# Check is_processing: false before continuing
```

### 3. Start Analysis

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What was the maximum altitude and were there any GPS issues?",
    "session_id": "abc123"
  }'
```

### 4. Advanced Analysis

```bash
# Anomaly detection
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find any anomalies in GPS, power, and attitude data",
    "session_id": "abc123"
  }'

# Flight phase analysis
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze the takeoff phase performance",
    "session_id": "abc123"
  }'

# Timeline of events
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me a timeline of all critical events during the flight",
    "session_id": "abc123"
  }'
```

## Advanced Analysis Tools

The V2 backend includes sophisticated analysis tools automatically available during chat:

### 1. **Anomaly Detection**
- Detects unusual patterns in GPS, attitude, power, and other systems
- Severity classification (Critical/Warning/Info)
- Temporal correlation analysis

### 2. **Flight Event Detection**
- GPS signal loss/recovery with timestamps
- Flight mode changes
- Critical system alerts
- Power issues and battery problems
- Attitude control problems

### 3. **Flight Phase Analysis**
- Takeoff, cruise, and landing phase identification
- Phase-specific performance metrics
- Stability and efficiency analysis

### 4. **Timeline Analysis**
- Chronological event sequence
- Multi-system correlation
- Issue progression tracking

### 5. **Custom Code Execution**
- Dynamic Python analysis
- Custom calculations and metrics
- Statistical analysis and comparisons

## ArduPilot Message Support

Built-in knowledge of 25+ ArduPilot message types including:

| Message | Description |
|---------|-------------|
| GPS | Position data - Lat, Lng, Alt, Speed, Satellite count |
| ATT | Attitude data - Roll, Pitch, Yaw from attitude controller |
| MODE | Flight mode changes with timestamps |
| CURR | Battery/Power data - Voltage, Current, Total consumption |
| IMU | Inertial measurement unit - Accelerometer and gyroscope data |
| BARO | Barometer readings - Altitude and pressure |
| MSG | System messages and alerts |
| PARM | Parameter values and configuration |
| RCIN/RCOUT | RC input/output channels |
| VIBE | Vibration levels affecting sensor performance |
| XKF1-4 | Extended Kalman Filter states and health |

## Performance Optimization

### High-Performance Parsing

The V2 backend includes multiple parsing strategies for optimal performance:

1. **Memory-Mapped Parsing** (< 50MB files)
   - Fastest for smaller files
   - Uses `mavutil` with batching
   - Low memory overhead

2. **Parallel Chunked Parsing** (> 50MB files)
   - Processes large files in parallel
   - Automatic load balancing
   - Scales with CPU cores

3. **Optimized Sequential** (fallback)
   - Enhanced batch processing
   - Efficient DataFrame creation
   - Smart dtype optimization

### Expected Performance

| File Size | Processing Time | Improvement over V1 |
|-----------|-----------------|-------------------|
| 5 MB      | 2-4 seconds     | 60-75% faster     |
| 20 MB     | 8-15 seconds    | 65-75% faster     |
| 50 MB     | 20-35 seconds   | 70-80% faster     |

## Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4
MAX_TOKENS=2000
TEMPERATURE=0.3

# Server Configuration
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:8080,http://localhost:3000

# Performance Configuration
USE_OPTIMIZED_PARSER=true
PARSER_MAX_WORKERS=0  # 0 = auto-detect CPU cores
```

## Testing

### Automated Tests

```bash
# Run comprehensive test suite
python test_new_backend.py
```

### Performance Testing

```bash
# Test parsing performance with your log files
python performance_test.py /path/to/your/flight_log.bin
```

### Manual Testing

```bash
# Test file upload
curl -X POST -F 'file=@test_flight.bin' http://localhost:8000/sessions/upload-log

# Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze this flight", "session_id": "your-session-id"}'
```

## Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t uav-chat-backend-v2 .

# Run container
docker run -p 8000:8000 --env-file .env uav-chat-backend-v2
```

### Security Considerations

- Keep OpenAI API key secure in environment variables
- Configure CORS origins appropriately for your domain
- Use HTTPS in production
- Implement rate limiting for file uploads
- Monitor API usage and costs
- Consider file size limits for uploads

## Development

### File Structure

```
backend/
├── main.py                    # FastAPI application with V2 endpoints
├── chat_service_v2.py         # Agentic chat service with tools
├── models.py                  # Pydantic data models for V2
├── config.py                  # Configuration management
├── optimized_parser.py        # High-performance log parsing
├── requirements.txt           # Python dependencies
├── start_server.py           # Server startup script
├── test_new_backend.py       # V2 test suite
├── performance_test.py       # Performance benchmarking
└── README.md                 # This file
```

### Adding New Analysis Tools

To add new analysis capabilities:

1. Add a new tool function in `chat_service_v2.py`
2. Register it in the `tools` array in the `chat()` method
3. Implement the tool logic following the existing pattern
4. Update documentation and tests

### Performance Monitoring

The backend logs detailed performance metrics:

```
[session_id] ⚡ OPTIMIZED PARSING COMPLETE! ⚡
[session_id] Processing time: 12.84s
[session_id] Total messages: 89,432
[session_id] Message types: 23
[session_id] Throughput: 6,967 messages/second
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   - Verify key is set in `.env` file
   - Check key has sufficient credits
   - Ensure key has access to GPT-4

2. **File Upload Issues**
   - Check file size limits
   - Verify .bin/.tlog file format
   - Monitor server logs for parsing errors

3. **Processing Timeouts**
   - Large files may take time to process
   - Check `is_processing` status via session endpoint
   - Monitor server resources

4. **Analysis Quality Issues**
   - Ensure log file contains relevant data
   - Check for complete flights (takeoff to landing)
   - Verify message types are supported

### Performance Tuning

For large files or high-volume usage:

```bash
# Increase parser workers
PARSER_MAX_WORKERS=8

# Monitor system resources
top -p $(pgrep -f uvicorn)

# Check memory usage
free -h
```

## License

This project is licensed under the same terms as the main UAV Log Viewer project. 