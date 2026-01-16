"""
FastAPI application entry point for the Audio Proctoring service.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import logging
import sys
import os

from app.config import API_V1_PREFIX
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Audio Proctoring - Speaker Detection API",
    description="""
## Anti-Cheat Audio Proctoring Service

This service analyzes audio to detect if someone other than the main speaker is speaking.

### How it works:

1. **Create a session** - Start a new proctoring session
2. **Submit first audio** - The speaker in this audio becomes the "main speaker"
3. **Submit subsequent audio** - Each audio is compared against the main speaker
4. **Detect anomalies** - If a different voice is detected, it's flagged as a foreign speaker

### Key Features:

- 🎙️ **Voice Fingerprinting** - Creates unique 256-dimensional embeddings for each voice
- 🔍 **Speaker Verification** - Compares voices using cosine similarity
- ⚠️ **Anomaly Detection** - Flags foreign speakers with detailed reports
- 📊 **Session Tracking** - Maintains speaker profiles throughout the session
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Include API routes
app.include_router(router, prefix=API_V1_PREFIX)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


# Root endpoint - redirect to UI
@app.get("/", tags=["Root"])
async def root():
    """Redirect to the web UI."""
    return RedirectResponse(url="/static/index.html")


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Audio Proctoring Service Starting...")
    logger.info("=" * 60)
    
    # Pre-load models (this takes a few seconds)
    logger.info("Loading speaker embedding model...")
    from app.services.speaker_embedding import get_embedding_service
    get_embedding_service()
    logger.info("Speaker embedding model loaded successfully!")
    
    logger.info("=" * 60)
    logger.info("Service is ready to accept requests!")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Audio Proctoring Service shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
