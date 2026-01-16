# Audio Proctoring - Speaker Detection Service

Anti-cheat audio proctoring service that detects if someone other than the main speaker is talking.

## Features

- 🎙️ **Voice Fingerprinting** - Creates unique 256-dimensional voice embeddings
- 🔍 **Speaker Verification** - Compares voices using cosine similarity
- ⚠️ **Foreign Speaker Detection** - Flags when a different person speaks
- 📊 **Session Management** - Maintains speaker profiles throughout the session

## Quick Start

### 1. Install Dependencies

```powershell
cd audio-proctoring
pip install -r requirements.txt
```

### 2. Run the Server

```powershell
uvicorn app.main:app --reload --port 8000
```

### 3. Test the API

Open http://localhost:8000/docs for the interactive API documentation.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/session` | POST | Create a new proctoring session |
| `/api/v1/session/{id}` | GET | Get session status |
| `/api/v1/session/{id}` | DELETE | End and delete session |
| `/api/v1/session/{id}/analyze` | POST | Analyze audio chunk |
| `/api/v1/health` | GET | Health check |

## How It Works

1. **Create Session**: Start a new proctoring session
2. **Register Main Speaker**: Submit first audio - this registers the "main speaker"
3. **Analyze Audio**: Submit subsequent audio chunks
   - If speaker matches main speaker → `is_anomaly: false`
   - If different speaker detected → `is_anomaly: true, anomaly_type: "foreign_speaker"`

## Example Usage

```python
import requests

# Create session
response = requests.post("http://localhost:8000/api/v1/session")
session_id = response.json()["session_id"]

# Analyze audio
with open("audio.wav", "rb") as f:
    response = requests.post(
        f"http://localhost:8000/api/v1/session/{session_id}/analyze",
        files={"audio": f}
    )
    result = response.json()
    
    if result["is_anomaly"]:
        print("⚠️ Foreign speaker detected!")
    else:
        print("✓ Main speaker verified")
```

## Configuration

Edit `app/config.py` to customize:

- `SIMILARITY_THRESHOLD` - Sensitivity for speaker matching (default: 0.75)
- `MIN_AUDIO_DURATION` - Minimum audio length (default: 2 seconds)
- `MAX_AUDIO_DURATION` - Maximum audio length (default: 60 seconds)
- `SESSION_TIMEOUT` - Session expiry time (default: 1 hour)

## Project Structure

```
audio-proctoring/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── main.py                # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── audio_processor.py    # Audio loading & VAD
│       ├── speaker_embedding.py  # Voice fingerprinting
│       ├── session_manager.py    # Session tracking
│       └── anomaly_detector.py   # Foreign speaker detection
├── test_poc.py                # Test script
├── requirements.txt           # Dependencies
└── README.md
```

## Testing

```powershell
# Run test script
python test_poc.py
```

Add test audio files to `test_audio/` directory:
- `speaker1_sample1.wav` - Main speaker (for registration)
- `speaker1_sample2.wav` - Same speaker (for verification)
- `speaker2_sample1.wav` - Different speaker (for anomaly detection)
