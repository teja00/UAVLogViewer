# UAV Log Viewer Chatbot Backend

A FastAPI-based backend service that provides intelligent chatbot functionality for analyzing UAV telemetry logs. This backend integrates with OpenAI's GPT models to provide natural language querying of MAVLink flight data.

## Features

### 🚁 Telemetry Analysis
- Parses UAV flight logs (.bin, .tlog files)
- Extracts flight data including GPS tracks, attitude, battery status, and flight modes
- Provides statistical analysis of flight performance

### 🤖 Intelligent Chatbot
- Natural language querying of flight data
- Context-aware conversations using OpenAI GPT models
- Maintains conversation history and session state
- Proactive clarification when needed

### 📊 Data Insights
- Flight statistics (duration, altitude, speed, etc.)
- Battery performance analysis
- GPS trajectory and position data
- Event detection and timeline analysis

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd UAVLogViewer/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env file and add your OpenAI API key
   ```

5. **Start the server**:
   ```bash
   python start_server.py
   ```

   Or run directly with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
MAX_TOKENS=2000
TEMPERATURE=0.7
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info
```

### API Configuration
- **Host**: `0.0.0.0` (listens on all interfaces)
- **Port**: `8000` (default)
- **CORS**: Configured for frontend origins
- **File Upload**: Max 100MB, supports .bin, .tlog, .txt files

## API Endpoints

### Health Check
```http
GET /health
```
Returns server status and configuration info.

### File Upload
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Binary file (.bin, .tlog, .txt)
- session_id: Optional session identifier
```

Uploads and parses a telemetry file, returning parsing results and session info.

### Chat
```http
POST /chat
Content-Type: application/json

{
  "message": "What was the maximum altitude during the flight?",
  "session_id": "session-uuid",
  "file_data": {} // optional additional data
}
```

Sends a message to the chatbot and returns an intelligent response based on the uploaded telemetry data.

### Session Management

#### Create Session
```http
POST /sessions
```

#### Get Session Info
```http
GET /sessions/{session_id}
```

#### Get Conversation History
```http
GET /sessions/{session_id}/history
```

#### Clear Session
```http
DELETE /sessions/{session_id}
```

#### List All Sessions
```http
GET /sessions
```

### Telemetry Analysis
```http
POST /telemetry/analyze
Content-Type: application/json

{
  "query": "altitude analysis",
  "messages": {
    // telemetry message data structure
  }
}
```

Direct telemetry data analysis endpoint for custom queries.

## Usage Examples

### Basic Chat Flow

1. **Upload a flight log**:
   ```bash
   curl -X POST "http://localhost:8000/upload" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@flight_log.bin"
   ```

2. **Start chatting**:
   ```bash
   curl -X POST "http://localhost:8000/chat" \
        -H "Content-Type: application/json" \
        -d '{
          "message": "What was the total flight time?",
          "session_id": "your-session-id"
        }'
   ```

### Example Questions

The chatbot can answer questions like:
- "What was the highest altitude reached during the flight?"
- "When did the GPS signal first get lost?"
- "What was the maximum battery temperature?"
- "How long was the total flight time?"
- "List all critical errors that happened mid-flight."
- "When was the first instance of RC signal loss?"

## Architecture

### Core Components

1. **FastAPI Application** (`main.py`)
   - REST API endpoints
   - Request/response handling
   - CORS configuration

2. **Chat Service** (`chat_service.py`)
   - OpenAI API integration
   - Conversation management
   - Context formatting

3. **Telemetry Parser** (`telemetry_parser.py`)
   - Flight data extraction
   - Statistical analysis
   - Query processing

4. **Data Models** (`models.py`)
   - Pydantic schemas
   - Type definitions
   - Validation

### Data Flow

```
Upload File → Parse Telemetry → Store in Session → Chat Query → 
OpenAI API → Context + Telemetry Data → Intelligent Response
```

## Integration with Frontend

The backend is designed to integrate seamlessly with the existing Vue.js frontend:

1. **File Upload**: Replace or extend existing file upload to also send data to backend
2. **Chat Interface**: Add chat UI components that communicate with `/chat` endpoint
3. **Session Management**: Maintain session IDs between frontend and backend
4. **Data Sharing**: Backend can receive parsed telemetry data from existing frontend parsers

## Development

### Adding New Features

1. **New Analysis Functions**: Add methods to `TelemetryParser` class
2. **Chat Capabilities**: Extend system prompts and context formatting in `ChatService`
3. **API Endpoints**: Add new routes in `main.py`
4. **Data Models**: Define new Pydantic models in `models.py`

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### API Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_server.py"]
```

### Production Considerations

- Use environment variables for all configuration
- Set up proper logging and monitoring
- Configure rate limiting for API endpoints
- Use HTTPS in production
- Consider using Redis for session storage in multi-instance deployments

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Working**
   - Verify the API key is correct
   - Check if you have sufficient credits
   - Ensure the key has proper permissions

2. **File Upload Fails**
   - Check file size (max 100MB)
   - Verify file format (.bin, .tlog, .txt)
   - Ensure proper Content-Type headers

3. **CORS Issues**
   - Update `ALLOWED_ORIGINS` in `config.py`
   - Ensure frontend is running on allowed origin

### Logs and Debugging

- Server logs show detailed error information
- Use `LOG_LEVEL=debug` for verbose logging
- Check `/health` endpoint for system status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is part of the UAV Log Viewer application. See the main repository for license information. 