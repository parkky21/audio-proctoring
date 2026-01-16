"""
API routes for the audio proctoring service.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from datetime import datetime
import logging

from app.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    SessionStatusResponse,
    DeleteSessionResponse,
    AnalyzeAudioResponse,
    HealthCheckResponse,
    ErrorResponse,
    AllSessionsResponse,
)
from app.services.session_manager import get_session_manager
from app.services.anomaly_detector import get_anomaly_detector

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Health Check ==============

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """Check if the service is running."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


# ============== Session Management ==============

@router.post(
    "/session",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Session"],
    summary="Create a new proctoring session"
)
async def create_session(request: CreateSessionRequest = None):
    """
    Create a new proctoring session.
    
    The first audio submitted to this session will register the main speaker.
    Subsequent audio will be compared against the main speaker.
    """
    session_manager = get_session_manager()
    
    session_id = request.session_id if request else None
    session = session_manager.create_session(session_id)
    
    logger.info(f"Created session: {session.session_id}")
    
    return CreateSessionResponse(
        session_id=session.session_id,
        message="Session created successfully. Submit the first audio to register the main speaker.",
        created_at=session.created_at.isoformat()
    )


@router.get(
    "/session/{session_id}",
    response_model=SessionStatusResponse,
    tags=["Session"],
    summary="Get session status"
)
async def get_session_status(session_id: str):
    """Get the current status of a proctoring session."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or has expired"
        )
    
    return SessionStatusResponse(**session.to_dict())


@router.delete(
    "/session/{session_id}",
    response_model=DeleteSessionResponse,
    tags=["Session"],
    summary="Delete a proctoring session"
)
async def delete_session(session_id: str):
    """
    Delete a proctoring session and clean up resources.
    
    This should be called when the proctoring session ends.
    """
    session_manager = get_session_manager()
    deleted = session_manager.delete_session(session_id)
    
    return DeleteSessionResponse(
        session_id=session_id,
        message="Session deleted successfully" if deleted else "Session not found",
        deleted=deleted
    )


@router.get(
    "/sessions",
    response_model=AllSessionsResponse,
    tags=["Session"],
    summary="List all active sessions"
)
async def list_all_sessions():
    """List all active proctoring sessions (admin endpoint)."""
    session_manager = get_session_manager()
    sessions = session_manager.get_all_sessions()
    
    return AllSessionsResponse(
        total_sessions=len(sessions),
        sessions=[SessionStatusResponse(**s) for s in sessions]
    )


# ============== Registration ==============

@router.post(
    "/session/{session_id}/start-registration",
    tags=["Registration"],
    summary="Start speaker registration phase"
)
async def start_registration(session_id: str):
    """
    Start the speaker registration phase.
    
    This begins collecting speech audio to create a speaker reference.
    - Audio chunks are accumulated until 5 seconds of speech is collected
    - Only actual speech (not silence) counts toward the 5 seconds
    - Call /analyze endpoint after this to submit audio chunks
    
    The registration will complete automatically when enough speech is collected.
    """
    from app.api.schemas import StartRegistrationResponse
    from app.services.session_manager import REGISTRATION_TARGET_DURATION
    
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    if session.is_main_speaker_set:
        return StartRegistrationResponse(
            session_id=session_id,
            success=False,
            message="Main speaker is already registered. Use a new session to register a different speaker.",
            target_duration=REGISTRATION_TARGET_DURATION
        )
    
    success = session_manager.start_registration(session_id)
    
    if success:
        logger.info(f"Registration started for session {session_id}")
        return StartRegistrationResponse(
            session_id=session_id,
            success=True,
            message=f"Registration started. Speak clearly to register your voice ({REGISTRATION_TARGET_DURATION:.0f}s of speech needed).",
            target_duration=REGISTRATION_TARGET_DURATION
        )
    else:
        return StartRegistrationResponse(
            session_id=session_id,
            success=False,
            message="Failed to start registration",
            target_duration=REGISTRATION_TARGET_DURATION
        )


# ============== Audio Analysis ==============

@router.post(
    "/session/{session_id}/analyze",
    response_model=AnalyzeAudioResponse,
    tags=["Analysis"],
    summary="Analyze audio for speaker verification"
)
async def analyze_audio(
    session_id: str,
    audio: UploadFile = File(..., description="Audio file to analyze (WAV, MP3, etc.)")
):
    """
    Analyze an audio chunk for speaker verification.
    
    **First audio**: Registers the speaker as the main speaker for the session.
    
    **Subsequent audio**: Compares the speaker against the main speaker.
    - If the speaker matches → returns is_anomaly=false
    - If a different speaker is detected → returns is_anomaly=true with anomaly_type="foreign_speaker"
    
    Supported formats: WAV, MP3, FLAC, OGG, and other formats supported by librosa.
    """
    # Check session exists
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found. Please create a session first."
        )
    
    # Read audio bytes
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file"
            )
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {str(e)}"
        )
    
    # Analyze audio
    detector = get_anomaly_detector()
    result = detector.analyze_audio_bytes(session_id, audio_bytes)
    
    logger.info(
        f"Analysis complete for session {session_id}: "
        f"is_main_speaker={result.is_main_speaker}, "
        f"similarity={result.similarity_score:.4f}, "
        f"anomaly={result.is_anomaly}"
    )
    
    return AnalyzeAudioResponse(**result.to_dict())


# ============== Utility Endpoints ==============

@router.post(
    "/cleanup",
    tags=["Admin"],
    summary="Clean up expired sessions"
)
async def cleanup_expired_sessions():
    """Remove all expired sessions to free up memory."""
    session_manager = get_session_manager()
    count = session_manager.cleanup_expired_sessions()
    
    return {
        "message": f"Cleaned up {count} expired sessions",
        "sessions_removed": count
    }
