"""
Speaker embedding service using Resemblyzer for voice fingerprinting.
"""
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from typing import Optional, Tuple
import logging

from app.config import SAMPLE_RATE, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class SpeakerEmbeddingService:
    """
    Handles speaker embedding extraction and comparison using Resemblyzer.
    
    Resemblyzer uses a neural network trained on speaker verification to create
    256-dimensional embeddings that capture unique voice characteristics.
    """

    def __init__(self):
        logger.info("Initializing SpeakerEmbeddingService...")
        self.encoder = VoiceEncoder()
        self.embedding_dim = 256
        logger.info("SpeakerEmbeddingService initialized successfully")

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Extract a speaker embedding from audio.

        Args:
            audio: Audio array (mono, float32)
            sample_rate: Sample rate of the audio

        Returns:
            256-dimensional embedding vector

        Raises:
            ValueError: If audio is too short or invalid
        """
        try:
            # Preprocess audio for Resemblyzer
            # This handles normalization and ensures correct format
            processed_audio = preprocess_wav(audio, source_sr=sample_rate)

            if len(processed_audio) < sample_rate:  # Less than 1 second
                logger.warning("Audio is very short, embedding quality may be reduced")

            # Extract embedding
            embedding = self.encoder.embed_utterance(processed_audio)

            logger.debug(f"Extracted embedding with shape {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise ValueError(f"Failed to extract speaker embedding: {e}")

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            logger.warning("Zero norm embedding detected")
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp to [0, 1] range (cosine similarity can be negative for very different voices)
        similarity = max(0.0, min(1.0, float(similarity)))

        logger.debug(f"Calculated similarity: {similarity:.4f}")
        return similarity

    def is_same_speaker(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Tuple[bool, float]:
        """
        Determine if two embeddings are from the same speaker.

        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            threshold: Similarity threshold for same speaker determination

        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        similarity = self.calculate_similarity(embedding1, embedding2)
        is_same = similarity >= threshold

        logger.info(f"Speaker comparison: similarity={similarity:.4f}, threshold={threshold}, same_speaker={is_same}")

        return is_same, similarity

    def average_embeddings(self, embeddings: list) -> np.ndarray:
        """
        Calculate the average of multiple embeddings.

        This is useful for creating a more robust speaker profile
        from multiple audio samples.

        Args:
            embeddings: List of embedding arrays

        Returns:
            Averaged embedding vector
        """
        if not embeddings:
            raise ValueError("Cannot average empty list of embeddings")

        stacked = np.stack(embeddings)
        averaged = np.mean(stacked, axis=0)

        # Re-normalize the averaged embedding
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return averaged


# Singleton instance
_embedding_service: Optional[SpeakerEmbeddingService] = None


def get_embedding_service() -> SpeakerEmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = SpeakerEmbeddingService()
    return _embedding_service
