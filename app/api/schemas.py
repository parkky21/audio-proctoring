"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============== Session Schemas ==============

class CreateSessionRequest(BaseModel):
    """Request to create a new proctoring session."""
    session_id: Optional[str] = Field(
        None,
        description="Optional custom session ID. If not provided, a UUID will be generated."
    )


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""
    session_id: str
    message: str
    created_at: str


class SessionStatusResponse(BaseModel):
    """Response with session status details."""
    session_id: str
    is_main_speaker_set: bool
    created_at: str
    last_activity: str
    anomaly_count: int
    total_audio_chunks: int
    embedding_samples_count: int


class DeleteSessionResponse(BaseModel):
    """Response after deleting a session."""
    session_id: str
    message: str
    deleted: bool


# ============== Analysis Schemas ==============

class AnalyzeAudioResponse(BaseModel):
    """Response after analyzing an audio chunk."""
    session_id: str
    is_main_speaker: bool = Field(
        description="Whether the speaker in the audio matches the main speaker"
    )
    similarity_score: float = Field(
        description="Cosine similarity score between 0 and 1"
    )
    is_anomaly: bool = Field(
        description="Whether an anomaly was detected"
    )
    anomaly_type: str = Field(
        description="Type of anomaly: none, foreign_speaker, overlapping_speech, whisper_detected, multiple_anomalies, no_speech, audio_too_short, audio_too_long, processing_error"
    )
    message: str = Field(
        description="Human-readable message about the analysis result"
    )
    audio_duration: float = Field(
        description="Duration of the analyzed audio in seconds"
    )
    is_first_audio: bool = Field(
        False,
        description="Whether this was the first audio that registered the main speaker"
    )
    # New fields for enhanced detection
    is_whisper: bool = Field(
        False,
        description="Whether whispered speech was detected"
    )
    whisper_confidence: float = Field(
        0.0,
        description="Confidence level for whisper detection (0-1)"
    )
    has_overlapping_speech: bool = Field(
        False,
        description="Whether overlapping speech (multiple people talking) was detected"
    )
    overlap_confidence: float = Field(
        0.0,
        description="Confidence level for overlapping speech detection (0-1)"
    )
    anomalies_detected: List[str] = Field(
        default_factory=list,
        description="List of all anomalies detected in the audio"
    )


# ============== Health Check Schemas ==============

class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    timestamp: str
    version: str = "1.0.0"


# ============== Error Schemas ==============

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    session_id: Optional[str] = None


# ============== Admin Schemas ==============

class AllSessionsResponse(BaseModel):
    """Response listing all active sessions."""
    total_sessions: int
    sessions: List[SessionStatusResponse]
