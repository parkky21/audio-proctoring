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
    REGISTRATION_IN_PROGRESS = "registration_in_progress"  # New: registration phase


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
    # Registration phase fields
    in_registration: bool = False
    registration_progress: float = 0.0
    registration_speech_collected: float = 0.0
    registration_target_duration: float = 5.0
    registration_complete: bool = False

    def __post_init__(self):
        if self.anomalies_detected is None:
            self.anomalies_detected = []

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        result = {
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
        # Add registration fields when in registration phase
        if self.in_registration or self.registration_complete:
            result.update({
                "in_registration": self.in_registration,
                "registration_progress": round(self.registration_progress, 4),
                "registration_speech_collected": round(self.registration_speech_collected, 2),
                "registration_target_duration": self.registration_target_duration,
                "registration_complete": self.registration_complete,
            })
        return result


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
        
        Also handles registration phase for accumulating speaker audio.
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
            
            # Handle registration phase - accumulate speech for reference
            if session.in_registration_phase:
                return self._handle_registration_phase(
                    session_id, audio, sample_rate, duration, audio_analysis
                )
            
            # Check if speech was found (for non-registration mode)
            if not audio_analysis.has_speech:
                # During normal detection, no speech is just skipped (not an anomaly)
                if session.is_main_speaker_set:
                    return AnalysisResult(
                        session_id=session_id,
                        is_main_speaker=True,  # Assume main speaker is present
                        similarity_score=1.0,
                        is_anomaly=False,
                        anomaly_type=AnomalyType.NONE,
                        message="No speech detected - skipping analysis",
                        audio_duration=duration,
                    )
                # Not in registration and no main speaker - prompt to start registration
                return AnalysisResult(
                    session_id=session_id,
                    is_main_speaker=False,
                    similarity_score=0.0,
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NO_SPEECH,
                    message="No speech detected. Start registration to enroll speaker.",
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

            # Check if main speaker is set (legacy flow - immediate registration)
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

    def _handle_registration_phase(
        self,
        session_id: str,
        audio: np.ndarray,
        sample_rate: int,
        duration: float,
        audio_analysis: AudioAnalysis
    ) -> AnalysisResult:
        """
        Handle audio during registration phase - accumulate speech segments.
        
        Only speech portions are accumulated toward the target duration.
        When enough speech is collected, registration is completed.
        """
        session = self.session_manager.get_session(session_id)
        if session is None:
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message="Session not found",
                audio_duration=duration,
            )
        
        # Get current registration progress
        progress_info = self.session_manager.get_registration_progress(session_id)
        
        # Log energy levels for debugging
        logger.info(
            f"Registration audio analysis - Energy: {audio_analysis.avg_energy:.4f}, "
            f"Segments: {len(audio_analysis.speech_segments)}, "
            f"Has speech: {audio_analysis.has_speech}"
        )
        
        # Check for actual speech - require BOTH detected segments AND sufficient energy
        # Typical speech RMS is 0.1-0.3, noise is usually below 0.03
        MIN_REGISTRATION_ENERGY = 0.08  # Lowered from 0.08 to catch more speech
        has_valid_speech = (
            audio_analysis.has_speech 
            and audio_analysis.speech_segments 
            and audio_analysis.avg_energy >= MIN_REGISTRATION_ENERGY
        )
        
        # Calculate speech duration from detected speech segments
        speech_duration = 0.0
        if has_valid_speech:
            for start, end in audio_analysis.speech_segments:
                speech_duration += (end - start)
        
        logger.info(
            f"Valid speech check - Energy OK: {audio_analysis.avg_energy >= MIN_REGISTRATION_ENERGY}, "
            f"Speech duration: {speech_duration:.2f}s"
        )
        
        # If no valid speech in this chunk, return progress without adding
        if speech_duration < 0.5:  # Require at least 500ms of clear speech per chunk
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=False,
                anomaly_type=AnomalyType.REGISTRATION_IN_PROGRESS,
                message=f"🎤 Speak clearly into the microphone... ({progress_info['speech_collected']:.1f}s / {progress_info['target_duration']:.0f}s)",
                audio_duration=duration,
                in_registration=True,
                registration_progress=progress_info['progress'],
                registration_speech_collected=progress_info['speech_collected'],
                registration_target_duration=progress_info['target_duration'],
                registration_complete=False,
            )
        
        # Extract only speech portions from audio
        processed_audio = self.audio_processor.preprocess_audio(audio)
        
        # Add speech to registration buffer
        add_result = self.session_manager.add_registration_audio(
            session_id, processed_audio, speech_duration
        )
        
        if "error" in add_result:
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=False,
                similarity_score=0.0,
                is_anomaly=True,
                anomaly_type=AnomalyType.PROCESSING_ERROR,
                message=add_result["error"],
                audio_duration=duration,
            )
        
        # Check if registration is complete
        if add_result["complete"]:
            # Get combined audio and extract embedding
            combined_audio = self.session_manager.get_registration_audio(session_id)
            if combined_audio is not None and len(combined_audio) > 0:
                try:
                    embedding = self.embedding_service.extract_embedding(combined_audio, sample_rate)
                    success = self.session_manager.complete_registration(session_id, embedding)
                    
                    if success:
                        logger.info(f"Registration completed for session {session_id}")
                        return AnalysisResult(
                            session_id=session_id,
                            is_main_speaker=True,
                            similarity_score=1.0,
                            is_anomaly=False,
                            anomaly_type=AnomalyType.NONE,
                            message=f"✅ Main speaker registered! Collected {add_result['speech_collected']:.1f}s of speech. Ready for detection.",
                            audio_duration=duration,
                            is_first_audio=True,
                            in_registration=False,
                            registration_progress=1.0,
                            registration_speech_collected=add_result['speech_collected'],
                            registration_target_duration=add_result['target_duration'],
                            registration_complete=True,
                        )
                except Exception as e:
                    logger.error(f"Failed to extract embedding from registration audio: {e}")
                    return AnalysisResult(
                        session_id=session_id,
                        is_main_speaker=False,
                        similarity_score=0.0,
                        is_anomaly=True,
                        anomaly_type=AnomalyType.PROCESSING_ERROR,
                        message=f"Failed to register speaker: {str(e)}",
                        audio_duration=duration,
                    )
        
        # Registration still in progress
        return AnalysisResult(
            session_id=session_id,
            is_main_speaker=False,
            similarity_score=0.0,
            is_anomaly=False,
            anomaly_type=AnomalyType.REGISTRATION_IN_PROGRESS,
            message=f"🎙️ Registering... {add_result['progress']:.0%} ({add_result['speech_collected']:.1f}s / {add_result['target_duration']:.0f}s)",
            audio_duration=duration,
            in_registration=True,
            registration_progress=add_result['progress'],
            registration_speech_collected=add_result['speech_collected'],
            registration_target_duration=add_result['target_duration'],
            registration_complete=False,
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
        
        # Check if there's enough speech/energy to make a valid comparison
        # Skip comparison ONLY for very low energy (silence) to avoid false alerts
        # This should match or be slightly above the speech detection threshold
        MIN_COMPARISON_ENERGY = 0.06  # Matched to speech detection threshold
        
        logger.info(
            f"Comparison energy check - Energy: {audio_analysis.avg_energy:.4f}, "
            f"Threshold: {MIN_COMPARISON_ENERGY}, Pass: {audio_analysis.avg_energy >= MIN_COMPARISON_ENERGY}"
        )
        
        if audio_analysis.avg_energy < MIN_COMPARISON_ENERGY:
            logger.info(f"Skipping comparison - pure silence detected")
            return AnalysisResult(
                session_id=session_id,
                is_main_speaker=True,  # Assume main speaker when no speech
                similarity_score=1.0,
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                message="🔇 No speech detected - monitoring continues",
                audio_duration=duration,
            )
        
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
