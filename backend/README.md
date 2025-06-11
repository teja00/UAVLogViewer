# UAV Log Analyzer - AI Chat Backend

An OpenAI-powered chatbot backend for analyzing UAV flight telemetry data. This backend receives pre-parsed telemetry data from the frontend and provides intelligent analysis using GPT models with comprehensive ArduPilot knowledge.

## Architecture

The backend is designed to work with the existing Vue.js frontend that already handles:
- Binary log file parsing (.bin, .tlog, .txt files)
- Data extraction using JavaScript workers
- Real-time telemetry visualization

The backend focuses solely on:
- Receiving parsed telemetry data from frontend
- Providing AI-powered analysis using OpenAI GPT
- Maintaining conversation context and history
- Leveraging ArduPilot documentation knowledge

## Key Features

- **Frontend Integration**: Designed to work with existing Vue.js app's parsing logic
- **ArduPilot Knowledge**: Built-in understanding of 25+ ArduPilot message types
- **Conversation Management**: Maintains context across multiple questions
- **Intelligent Analysis**: Provides insights on flight performance, safety, and optimization
- **Session Management**: Tracks conversation history and telemetry data
- **Real-time Response**: Fast API responses for interactive chat experience

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
# Start the server
python start_server.py

# Or run directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the Backend

```bash
# Check health
curl http://localhost:8000/health

# Get simulated telemetry data for testing
curl -X POST http://localhost:8000/test/simulate-frontend-data
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check and configuration status
- `POST /chat` - Main chat endpoint (receives telemetry data + user message)
- `POST /sessions/{session_id}/telemetry` - Update session telemetry data
- `GET /sessions/{session_id}` - Get session information
- `DELETE /sessions/{session_id}` - Clear session history

### Testing Endpoints

- `POST /test/simulate-frontend-data` - Generate sample telemetry data for testing

## Data Flow

```
Frontend (Vue.js)           Backend (FastAPI + OpenAI)
┌─────────────────┐        ┌──────────────────────────┐
│ 1. User uploads │        │                          │
│    .bin file    │        │                          │
└─────────────────┘        │                          │
         │                 │                          │
┌─────────────────┐        │                          │
│ 2. JavaScript   │        │                          │
│    worker parses│        │                          │
│    telemetry    │        │                          │
└─────────────────┘        │                          │
         │                 │                          │
┌─────────────────┐        │ ┌──────────────────────┐ │
│ 3. Send parsed  │───────▶│ │ POST /chat           │ │
│    data + user  │        │ │ - telemetry_data     │ │
│    question     │        │ │ - message            │ │
└─────────────────┘        │ │ - session_id         │ │
         │                 │ └──────────────────────┘ │
┌─────────────────┐        │           │              │
│ 5. Display AI   │◀───────│ ┌──────────────────────┐ │
│    response     │        │ │ 4. OpenAI Analysis   │ │
└─────────────────┘        │ │ - ArduPilot knowledge│ │
                           │ │ - Context-aware      │ │
                           │ │ - Flight analysis    │ │
                           │ └──────────────────────┘ │
                           └──────────────────────────┘
```

## Example Usage

### 1. Frontend Sends Parsed Data

The frontend worker parses a .bin file and sends:

```javascript
const telemetryData = {
  messages: {
    GPS: {
      time_boot_ms: [1000, 2000, 3000],
      lat: [40.7128, 40.7129, 40.7130],
      lng: [-74.0060, -74.0061, -74.0062],
      alt: [10.5, 12.3, 15.8]
    },
    ATT: {
      time_boot_ms: [1000, 2000, 3000],
      Roll: [0.1, 0.2, -0.1],
      Pitch: [0.05, -0.1, 0.15],
      Yaw: [1.57, 1.58, 1.56]
    },
    MODE: {
      time_boot_ms: [500, 2500],
      asText: ["STABILIZE", "LOITER"]
    }
  },
  metadata: {
    startTime: 1640995200000,
    vehicleType: "Quadcopter",
    logType: "bin"
  }
};

// Send to backend
fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "What was the maximum altitude during this flight?",
    telemetry_data: telemetryData,
    session_id: "optional-session-id"
  })
});
```

### 2. Backend Response

```json
{
  "response": "Based on the GPS data in your flight log, the maximum altitude reached was 15.8 meters, occurring at timestamp 3000ms (3 seconds into the flight). The aircraft showed a steady climb from 10.5m to 15.8m over the first 3 seconds, which indicates a controlled ascent.",
  "session_id": "abc123-session-id",
  "conversation_count": 2
}
```

## ArduPilot Message Support

The backend has built-in knowledge of 25+ ArduPilot message types including:

| Message | Description |
|---------|-------------|
| GPS | Position data - Lat, Lng, Alt, Speed |
| ATT | Attitude data - Roll, Pitch, Yaw |
| MODE | Flight mode changes |
| CURR | Battery/Power data - Voltage, Current |
| IMU | Inertial measurement unit data |
| BARO | Barometer readings |
| MSG | System messages and alerts |
| PARM | Parameter values |
| RCIN/RCOUT | RC input/output channels |
| VIBE | Vibration levels |
| XKF1-4 | Extended Kalman Filter data |

## Testing Without Frontend

Use the test endpoint to simulate frontend-parsed data:

```bash
# Get sample data structure
curl -X POST http://localhost:8000/test/simulate-frontend-data

# Test chat with simulated data
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze the battery performance",
    "telemetry_data": {
      "messages": {
        "CURR": {
          "time_boot_ms": [1000, 2000, 3000],
          "Volt": [12.6, 12.4, 12.2],
          "Curr": [2.1, 2.3, 2.5]
        }
      },
      "metadata": {
        "logType": "bin",
        "vehicleType": "Quadcopter"
      }
    }
  }'
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_MODEL` - GPT model to use (default: gpt-4)
- `MAX_TOKENS` - Response length limit (default: 1000)
- `TEMPERATURE` - Response creativity (default: 0.3)
- `CORS_ORIGINS` - Allowed origins (default: localhost)

### Example .env File

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4
MAX_TOKENS=1500
TEMPERATURE=0.3
CORS_ORIGINS=http://localhost:8080,http://localhost:3000
```

## Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t uav-chat-backend .

# Run container
docker run -p 8000:8000 --env-file .env uav-chat-backend
```

### Security Considerations

- Keep OpenAI API key secure
- Configure CORS origins appropriately
- Use HTTPS in production
- Implement rate limiting if needed
- Monitor API usage and costs

## Development

### File Structure

```
backend/
├── main.py              # FastAPI application
├── chat_service.py      # OpenAI integration & conversation management
├── models.py            # Pydantic data models
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── start_server.py      # Server startup script
└── README.md           # This file
```

### Adding New Message Types

To add support for new ArduPilot message types:

1. Update `ardupilot_messages` dict in `chat_service.py`
2. Add field definitions to `TelemetryMessage` model in `models.py`
3. Update context formatting in `_format_telemetry_context()`

### Testing

```bash
# Run basic health check
python -c "
import requests
r = requests.get('http://localhost:8000/health')
print(r.json())
"

# Test with simulated data
python -c "
import requests
data = requests.post('http://localhost:8000/test/simulate-frontend-data').json()
print('Sample data generated successfully')
"
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   - Verify key is set in `.env` file
   - Check key has sufficient credits
   - Ensure key has access to specified model

2. **CORS Issues**
   - Update `CORS_ORIGINS` in config
   - Check frontend is making requests from allowed origin

3. **Module Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

4. **Connection Issues**
   - Verify server is running on correct port
   - Check firewall settings
   - Ensure frontend is pointing to correct backend URL

### Logging

The backend includes comprehensive logging. Check logs for:
- Session creation and updates
- OpenAI API calls and responses
- Error details and stack traces
- Performance metrics

## License

This project is licensed under the same terms as the main UAV Log Viewer project. 