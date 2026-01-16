"""
Anomaly detector for identifying foreign speakers, overlapping speech, and whispers.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np
import logging

from app.services.audio_processor import get_audio_processor, AudioAnalysis
from app.services.speaker_embedding import get_embedding_service
from app.services.session_manager import get_session_manager
from app.config import SIMILARITY_THRESHOLD, MIN_AUDIO_DURATION, MAX_AUDIO_DURATION

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    NONE = "none"
    FOREIGN_SPEAKER = "foreign_speaker"
    OVERLAPPING_SPEECH = "overlapping_speech"
    WHISPER_DETECTED = "whisper_detected"
    MULTIPLE_ANOMALIES = "multiple_anomalies"
    NO_SPEECH = "no_speech"
    AUDIO_TOO_SHORT = "audio_too_short"
    AUDIO_TOO_LONG = "audio_too_long"
    PROCESSING_ERROR = "processing_error"


@dataclass
class AnalysisResult:
    """Result of audio analysis with all detection details."""
    
    session_id: str
    is_main_speaker: bool
    similarity_score: float
    is_anomaly: bool
    anomaly_type: AnomalyType
    message: str
    audio_duration: float
    is_first_audio: bool = False
    # New fields for enhanced detection
    is_whisper: bool = False
    whisper_confidence: float = 0.0
    has_overlapping_speech: bool = False
    overlap_confidence: float = 0.0
    anomalies_detected: List[str] = None

    def __post_init__(self):
        if self.anomalies_detected is None:
            self.anomalies_detected = []

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "session_id": self.session_id,
            "is_main_speaker": self.is_main_speaker,
            "similarity_score": round(self.similarity_score, 4),
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type.value,
            "message": self.message,
            "audio_duration": round(self.audio_duration, 2),
            "is_first_audio": self.is_first_audio,
            "is_whisper": self.is_whisper,
            "whisper_confidence": round(self.whisper_confidence, 4),
            "has_overlapping_speech": self.has_overlapping_speech,
            "overlap_confidence": round(self.overlap_confidence, 4),
            "anomalies_detected": self.anomalies_detected,
        }


class AnomalyDetector:
    """
    Detects foreign speakers, overlapping speech, and whispers in audio.
    
    Workflow:
    1. First audio in session -> registers as main speaker
    2. Subsequent audio -> compared against main speaker
    3. Check for overlapping speech (multiple people talking)
    4. Check for whispers (potential cheating indicator)
    5. Flag all detected anomalies
    """

    def __init__(self):
        self.audio_processor = get_audio_processor()
        self.embedding_service = get_embedding_service()
        self.session_manager = get_session_manager()
        self.similarity_threshold = SIMILARITY_THRESHOLD
        logger.info("AnomalyDetector initialized with enhanced detection")

    def analyze_audio_bytes(
        self,
        session_id: str,
        audio_bytes: bytes,
        update_embedding: bool = True
    ) -> AnalysisResult:
        """Analyze audio bytes for all types of anomalies."""
        try:
            audio, sr = self.audio_processor.load_audio_from_bytes(audio_bytes)
            return self.analyze_audio(session_id, audio, sr, update_embedding)
        except Exception as e:
            logger.error(f"Failed to process audio bytes: {e}")
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message=f"Failed to process audio: {str(e)}",
                audio_duration=0.0,
            )

    def analyze_audio_file(
        self,
        session_id: str,
        file_path: str,
        update_embedding: bool = True
    ) -> AnalysisResult:
        """Analyze audio file for all types of anomalies."""
        try:
            audio, sr = self.audio_processor.load_audio_from_file(file_path)
            return self.analyze_audio(session_id, audio, sr, update_embedding)
        except Exception as e:
            logger.error(f"Failed to process audio file: {e}")
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message=f"Failed to process audio file: {str(e)}",
                audio_duration=0.0,
            )

    def analyze_audio(
        self,
        session_id: str,
        audio: np.ndarray,
        sample_rate: int,
        update_embedding: bool = True
    ) -> AnalysisResult:
        """
        Main analysis method with comprehensive anomaly detection.

        Detects:
        - Foreign speakers (different voice from main speaker)
        - Overlapping speech (multiple people talking simultaneously)
        - Whispers (potential cheating indicator)
        """
        # Get session
        session = self.session_manager.get_session(session_id)
        if session is None:
            logger.error(f"Session {session_id} not found")
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message="Session not found. Please create a session first.",
                audio_duration=0.0,
            )

        # Increment audio count
        self.session_manager.increment_audio_count(session_id)

        # Check audio duration
        duration = self.audio_processor.get_audio_duration(audio, sample_rate)

        if duration < MIN_AUDIO_DURATION:
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.AUDIO_TOO_SHORT,
                message=f"Audio too short ({duration:.2f}s). Minimum required: {MIN_AUDIO_DURATION}s",
                audio_duration=duration,
            )

        if duration > MAX_AUDIO_DURATION:
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.AUDIO_TOO_LONG,
                message=f"Audio too long ({duration:.2f}s). Maximum allowed: {MAX_AUDIO_DURATION}s",
                audio_duration=duration,
            )

        try:
            # Perform comprehensive audio analysis
            audio_analysis = self.audio_processor.analyze_audio(audio)
            
            # Check if speech was found
            if not audio_analysis.has_speech:
                return AnalysisResult(
                    session_id=session_id,
                    is_main_speaker=False,
                    similarity_score=0.0,
                    is_anomaly=True,
                    anomaly_type=AnomalyType.NO_SPEECH,
                    message="No speech detected in audio",
                    audio_duration=duration,
                )

            # Extract embedding from preprocessed audio
            processed_audio = self.audio_processor.preprocess_audio(audio)
            embedding = self.embedding_service.extract_embedding(processed_audio, sample_rate)

            # Collect all detected anomalies
            anomalies = []
            
            # Check for overlapping speech
            if audio_analysis.has_overlapping_speech:
                anomalies.append(f"overlapping_speech ({audio_analysis.overlap_confidence:.0%})")
                logger.warning(
                    f"OVERLAPPING SPEECH detected in session {session_id}! "
                    f"Confidence: {audio_analysis.overlap_confidence:.2%}"
                )
            
            # Check for whisper
            if audio_analysis.is_whisper:
                anomalies.append(f"whisper ({audio_analysis.whisper_confidence:.0%})")
                logger.warning(
                    f"WHISPER detected in session {session_id}! "
                    f"Confidence: {audio_analysis.whisper_confidence:.2%}"
                )

            # Check if this is the first audio (main speaker registration)
            if not session.is_main_speaker_set:
                return self._register_main_speaker(
                    session_id, embedding, duration, audio_analysis, anomalies
                )

            # Compare with main speaker and combine with other anomalies
            return self._compare_with_main_speaker(
                session_id, embedding, duration, update_embedding, audio_analysis, anomalies
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message=f"Analysis failed: {str(e)}",
                audio_duration=duration,
            )

    def _register_main_speaker(
        self,
        session_id: str,
        embedding: np.ndarray,
        duration: float,
        audio_analysis: AudioAnalysis,
        anomalies: List[str]
    ) -> AnalysisResult:
        """Register the first audio as the main speaker."""
        
        # Warn if registering with anomalies (but still allow)
        warning_msg = ""
        if anomalies:
            warning_msg = f" Warning: {', '.join(anomalies)} detected during registration."
        
        success = self.session_manager.set_main_speaker(session_id, embedding)
        
        if success:
            logger.info(f"Main speaker registered for session {session_id}")
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=True,
                similarity_score=1.0,
                is_anomaly=len(anomalies) > 0,
                anomaly_type=AnomalyType.MULTIPLE_ANOMALIES if anomalies else AnomalyType.NONE,
                message=f"Main speaker registered successfully.{warning_msg} Subsequent audio will be compared against this speaker.",
                audio_duration=duration,
                is_first_audio=True,
                is_whisper=audio_analysis.is_whisper,
                whisper_confidence=audio_analysis.whisper_confidence,
                has_overlapping_speech=audio_analysis.has_overlapping_speech,
                overlap_confidence=audio_analysis.overlap_confidence,
                anomalies_detected=anomalies,
            )
        else:
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message="Failed to register main speaker",
                audio_duration=duration,
            )

    def _compare_with_main_speaker(
        self,
        session_id: str,
        embedding: np.ndarray,
        duration: float,
        update_embedding: bool,
        audio_analysis: AudioAnalysis,
        anomalies: List[str]
    ) -> AnalysisResult:
        """Compare audio embedding with the main speaker and check for all anomalies."""
        main_embedding = self.session_manager.get_main_speaker(session_id)
        
        if main_embedding is None:
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message="Main speaker embedding not found",
                audio_duration=duration,
            )

        # Calculate similarity
        is_same, similarity = self.embedding_service.is_same_speaker(
            main_embedding, embedding, self.similarity_threshold
        )

        # Check for foreign speaker
        if not is_same:
            anomalies.append(f"foreign_speaker (similarity: {similarity:.0%})")
            logger.warning(
                f"FOREIGN SPEAKER detected in session {session_id}! "
                f"Similarity: {similarity:.2%}, Threshold: {self.similarity_threshold:.2%}"
            )

        # Determine overall result
        has_anomaly = len(anomalies) > 0
        
        if has_anomaly:
            self.session_manager.increment_anomaly_count(session_id)
            
            # Determine primary anomaly type
            if len(anomalies) > 1:
                anomaly_type = AnomalyType.MULTIPLE_ANOMALIES
            elif not is_same:
                anomaly_type = AnomalyType.FOREIGN_SPEAKER
            elif audio_analysis.has_overlapping_speech:
                anomaly_type = AnomalyType.OVERLAPPING_SPEECH
            elif audio_analysis.is_whisper:
                anomaly_type = AnomalyType.WHISPER_DETECTED
            else:
                anomaly_type = AnomalyType.MULTIPLE_ANOMALIES
            
            # Build message
            messages = []
            if not is_same:
                messages.append(f"⚠️ Foreign speaker (similarity: {similarity:.0%})")
            if audio_analysis.has_overlapping_speech:
                messages.append(f"⚠️ Overlapping speech detected ({audio_analysis.overlap_confidence:.0%} confidence)")
            if audio_analysis.is_whisper:
                messages.append(f"⚠️ Whisper detected ({audio_analysis.whisper_confidence:.0%} confidence)")
            
            message = " | ".join(messages) if messages else "Anomaly detected"
            
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=is_same,
                similarity_score=similarity,
                is_anomaly=True,
                anomaly_type=anomaly_type,
                message=message,
                audio_duration=duration,
                is_whisper=audio_analysis.is_whisper,
                whisper_confidence=audio_analysis.whisper_confidence,
                has_overlapping_speech=audio_analysis.has_overlapping_speech,
                overlap_confidence=audio_analysis.overlap_confidence,
                anomalies_detected=anomalies,
            )
        else:
            # No anomalies - update embedding if enabled
            if update_embedding:
                self.session_manager.update_main_speaker(session_id, embedding)

            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=True,
                similarity_score=similarity,
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                message=f"✅ Speaker verified (similarity: {similarity:.0%})",
                audio_duration=duration,
                is_whisper=False,
                whisper_confidence=0.0,
                has_overlapping_speech=False,
                overlap_confidence=0.0,
                anomalies_detected=[],
            )


# Singleton instance
_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get or create the anomaly detector singleton."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector
