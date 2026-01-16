"""
Session manager for tracking speaker profiles across proctoring sessions.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import logging
import threading

from app.config import MAX_EMBEDDING_SAMPLES, SESSION_TIMEOUT

logger = logging.getLogger(__name__)

# Registration phase configuration
REGISTRATION_TARGET_DURATION: float = 5.0  # 5 seconds of speech needed


@dataclass
class SessionData:
    """Data structure for a proctoring session."""
    
    session_id: str
    main_speaker_embedding: Optional[np.ndarray] = None
    embedding_samples: List[np.ndarray] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    anomaly_count: int = 0
    total_audio_chunks: int = 0
    is_main_speaker_set: bool = False
    
    # Registration phase fields
    in_registration_phase: bool = False
    registration_speech_duration: float = 0.0
    registration_audio_chunks: List[np.ndarray] = field(default_factory=list)
    registration_target_duration: float = REGISTRATION_TARGET_DURATION

    def to_dict(self) -> dict:
        """Convert session data to dictionary for API responses."""
        return {
            "session_id": self.session_id,
            "is_main_speaker_set": self.is_main_speaker_set,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "anomaly_count": self.anomaly_count,
            "total_audio_chunks": self.total_audio_chunks,
            "embedding_samples_count": len(self.embedding_samples),
            "in_registration_phase": self.in_registration_phase,
            "registration_progress": min(1.0, self.registration_speech_duration / self.registration_target_duration) if self.in_registration_phase else 0.0,
        }


class SessionManager:
    """
    Manages proctoring sessions and speaker profiles.
    
    Thread-safe implementation for handling concurrent requests.
    """

    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        logger.info("SessionManager initialized")

    def create_session(self, session_id: Optional[str] = None) -> SessionData:
        """
        Create a new proctoring session.

        Args:
            session_id: Optional custom session ID, generates UUID if not provided

        Returns:
            New SessionData object
        """
        with self._lock:
            if session_id is None:
                session_id = str(uuid.uuid4())

            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, returning existing")
                return self.sessions[session_id]

            session = SessionData(session_id=session_id)
            self.sessions[session_id] = session
            logger.info(f"Created new session: {session_id}")
            return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionData if found, None otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                # Check for timeout
                elapsed = (datetime.now() - session.last_activity).total_seconds()
                if elapsed > SESSION_TIMEOUT:
                    logger.info(f"Session {session_id} has timed out, cleaning up")
                    del self.sessions[session_id]
                    return None
            return session

    def set_main_speaker(self, session_id: str, embedding: np.ndarray) -> bool:
        """
        Set the main speaker embedding for a session.

        Args:
            session_id: Session identifier
            embedding: Speaker embedding to set as main speaker

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                logger.error(f"Session {session_id} not found")
                return False

            session.main_speaker_embedding = embedding.copy()
            session.embedding_samples = [embedding.copy()]
            session.is_main_speaker_set = True
            session.last_activity = datetime.now()
            
            logger.info(f"Main speaker set for session {session_id}")
            return True

    def get_main_speaker(self, session_id: str) -> Optional[np.ndarray]:
        """
        Get the main speaker embedding for a session.

        Args:
            session_id: Session identifier

        Returns:
            Main speaker embedding if set, None otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session and session.main_speaker_embedding is not None:
                return session.main_speaker_embedding.copy()
            return None

    def update_main_speaker(self, session_id: str, new_embedding: np.ndarray) -> bool:
        """
        Update the main speaker embedding by adding a new sample and recalculating average.

        Args:
            session_id: Session identifier
            new_embedding: New embedding sample from confirmed main speaker

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                logger.error(f"Session {session_id} not found")
                return False

            # Add new sample
            session.embedding_samples.append(new_embedding.copy())

            # Keep only the last N samples
            if len(session.embedding_samples) > MAX_EMBEDDING_SAMPLES:
                session.embedding_samples.pop(0)

            # Recalculate average embedding
            stacked = np.stack(session.embedding_samples)
            averaged = np.mean(stacked, axis=0)
            
            # Normalize
            norm = np.linalg.norm(averaged)
            if norm > 0:
                averaged = averaged / norm

            session.main_speaker_embedding = averaged
            session.last_activity = datetime.now()

            logger.debug(f"Updated main speaker embedding for session {session_id} (samples: {len(session.embedding_samples)})")
            return True

    def increment_anomaly_count(self, session_id: str) -> int:
        """
        Increment the anomaly count for a session.

        Args:
            session_id: Session identifier

        Returns:
            New anomaly count, -1 if session not found
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return -1

            session.anomaly_count += 1
            session.last_activity = datetime.now()
            logger.warning(f"Anomaly count incremented for session {session_id}: {session.anomaly_count}")
            return session.anomaly_count

    def increment_audio_count(self, session_id: str) -> int:
        """
        Increment the total audio chunks processed for a session.

        Args:
            session_id: Session identifier

        Returns:
            New audio count, -1 if session not found
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return -1

            session.total_audio_chunks += 1
            session.last_activity = datetime.now()
            return session.total_audio_chunks

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and clean up resources.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Session {session_id} deleted")
                return True
            return False

    def get_all_sessions(self) -> List[dict]:
        """
        Get summary of all active sessions.

        Returns:
            List of session summaries
        """
        with self._lock:
            return [session.to_dict() for session in self.sessions.values()]

    # ============== Registration Phase Methods ==============

    def start_registration(self, session_id: str) -> bool:
        """
        Start the registration phase for collecting speaker audio.

        Args:
            session_id: Session identifier

        Returns:
            True if registration started successfully, False otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                logger.error(f"Session {session_id} not found")
                return False

            if session.is_main_speaker_set:
                logger.warning(f"Session {session_id} already has main speaker set")
                return False

            session.in_registration_phase = True
            session.registration_speech_duration = 0.0
            session.registration_audio_chunks = []
            session.last_activity = datetime.now()

            logger.info(f"Registration phase started for session {session_id}")
            return True

    def add_registration_audio(
        self, 
        session_id: str, 
        audio_chunk: np.ndarray, 
        speech_duration: float
    ) -> dict:
        """
        Add audio chunk to registration buffer.

        Args:
            session_id: Session identifier
            audio_chunk: Audio data containing speech
            speech_duration: Duration of speech in the chunk (seconds)

        Returns:
            Dict with registration progress info
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return {"error": "Session not found", "complete": False}

            if not session.in_registration_phase:
                return {"error": "Not in registration phase", "complete": False}

            # Add audio chunk
            session.registration_audio_chunks.append(audio_chunk.copy())
            session.registration_speech_duration += speech_duration
            session.last_activity = datetime.now()

            progress = min(1.0, session.registration_speech_duration / session.registration_target_duration)
            is_complete = session.registration_speech_duration >= session.registration_target_duration

            logger.info(
                f"Registration progress for session {session_id}: "
                f"{session.registration_speech_duration:.2f}s / {session.registration_target_duration:.2f}s "
                f"({progress:.0%})"
            )

            return {
                "progress": progress,
                "speech_collected": session.registration_speech_duration,
                "target_duration": session.registration_target_duration,
                "complete": is_complete,
                "chunks_collected": len(session.registration_audio_chunks)
            }

    def get_registration_audio(self, session_id: str) -> Optional[np.ndarray]:
        """
        Get the combined registration audio for embedding extraction.

        Args:
            session_id: Session identifier

        Returns:
            Combined audio array or None if not available
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None or not session.registration_audio_chunks:
                return None

            # Concatenate all audio chunks
            combined = np.concatenate(session.registration_audio_chunks)
            return combined

    def complete_registration(self, session_id: str, embedding: np.ndarray) -> bool:
        """
        Complete registration phase and set the main speaker embedding.

        Args:
            session_id: Session identifier
            embedding: Speaker embedding extracted from registration audio

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                logger.error(f"Session {session_id} not found")
                return False

            session.main_speaker_embedding = embedding.copy()
            session.embedding_samples = [embedding.copy()]
            session.is_main_speaker_set = True
            session.in_registration_phase = False
            session.registration_audio_chunks = []  # Clear to free memory
            session.last_activity = datetime.now()

            logger.info(
                f"Registration completed for session {session_id} "
                f"(collected {session.registration_speech_duration:.2f}s of speech)"
            )
            return True

    def cancel_registration(self, session_id: str) -> bool:
        """
        Cancel the registration phase and clear collected audio.

        Args:
            session_id: Session identifier

        Returns:
            True if cancelled, False if session not found
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return False

            session.in_registration_phase = False
            session.registration_speech_duration = 0.0
            session.registration_audio_chunks = []
            session.last_activity = datetime.now()

            logger.info(f"Registration cancelled for session {session_id}")
            return True

    def get_registration_progress(self, session_id: str) -> Optional[dict]:
        """
        Get the current registration progress.

        Args:
            session_id: Session identifier

        Returns:
            Dict with progress info or None if not in registration
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None

            if not session.in_registration_phase:
                return {
                    "in_registration": False,
                    "progress": 0.0,
                    "speech_collected": 0.0,
                    "target_duration": session.registration_target_duration,
                }

            progress = min(1.0, session.registration_speech_duration / session.registration_target_duration)
            return {
                "in_registration": True,
                "progress": progress,
                "speech_collected": session.registration_speech_duration,
                "target_duration": session.registration_target_duration,
                "chunks_collected": len(session.registration_audio_chunks),
            }

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            now = datetime.now()
            expired = []
            
            for session_id, session in self.sessions.items():
                elapsed = (now - session.last_activity).total_seconds()
                if elapsed > SESSION_TIMEOUT:
                    expired.append(session_id)

            for session_id in expired:
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")

            return len(expired)


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the session manager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
