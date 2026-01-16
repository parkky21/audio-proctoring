"""
Configuration settings for the Audio Proctoring service.
"""
from typing import Final

# Speaker similarity threshold (cosine similarity)
# Higher = stricter matching, Lower = more lenient
# Typical range: 0.70 - 0.85
SIMILARITY_THRESHOLD: Final[float] = 0.80  # Increased from 0.75 for stricter detection

# Minimum audio duration in seconds for reliable embedding extraction
# Resemblyzer works best with 5-30 seconds of speech
MIN_AUDIO_DURATION: Final[float] = 2.0

# Maximum audio duration in seconds to process
MAX_AUDIO_DURATION: Final[float] = 60.0

# Target sample rate for audio processing (Resemblyzer requirement)
SAMPLE_RATE: Final[int] = 16000

# Maximum number of embedding samples to keep for averaging
MAX_EMBEDDING_SAMPLES: Final[int] = 10

# Minimum speech duration in seconds to consider valid
MIN_SPEECH_DURATION: Final[float] = 1.0

# Session timeout in seconds (1 hour)
SESSION_TIMEOUT: Final[int] = 3600

# API Settings
API_V1_PREFIX: Final[str] = "/api/v1"
