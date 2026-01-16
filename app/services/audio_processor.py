"""
Audio processing utilities for loading, resampling, and preprocessing audio.
Includes detection for overlapping speech and whispers.
"""
import io
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

from app.config import (
    SAMPLE_RATE,
    MIN_SPEECH_DURATION,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysis:
    """Results of audio analysis including whisper and overlap detection."""
    audio: np.ndarray
    sample_rate: int
    duration: float
    has_speech: bool
    is_whisper: bool
    whisper_confidence: float
    has_overlapping_speech: bool
    overlap_confidence: float
    speech_segments: List[Tuple[float, float]]
    avg_energy: float
    spectral_flatness: float


class AudioProcessor:
    """Handles audio loading, preprocessing, and advanced detection."""

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        # Whisper detection thresholds - more lenient
        self.whisper_energy_threshold = 0.015
        self.whisper_spectral_flatness_threshold = 0.6
        # Overlapping speech thresholds - MUCH stricter to reduce false positives
        self.overlap_pitch_threshold = 0.35  # Was 0.2, now need 35% of frames
        self.overlap_confidence_threshold = 0.6  # Need higher overall confidence

    def load_audio_from_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file and resample to target sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio from {file_path}: {len(audio)/sr:.2f}s at {sr}Hz")
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio from {file_path}: {e}")
            raise

    def load_audio_from_bytes(self, audio_bytes: bytes, original_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio from bytes - handles various formats including browser webm."""
        errors = []
        
        # Try direct BytesIO loading
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio from bytes (direct): {len(audio)/sr:.2f}s")
            return audio, sr
        except Exception as e:
            errors.append(f"Direct: {e}")
        
        # Try soundfile
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_file)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            logger.info(f"Loaded audio from bytes (soundfile): {len(audio)/self.sample_rate:.2f}s")
            return audio.astype(np.float32), self.sample_rate
        except Exception as e:
            errors.append(f"Soundfile: {e}")
        
        # Try temp files with different extensions
        for ext in ['.webm', '.ogg', '.wav', '.mp3']:
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                try:
                    audio, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
                    logger.info(f"Loaded audio from bytes (temp {ext}): {len(audio)/sr:.2f}s")
                    return audio, sr
                finally:
                    try: os.unlink(tmp_path)
                    except: pass
            except Exception as e:
                errors.append(f"Temp {ext}: {e}")
        
        # Try raw PCM
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) > self.sample_rate:
                logger.info(f"Loaded audio from bytes (raw PCM): {len(audio)/self.sample_rate:.2f}s")
                return audio, self.sample_rate
        except Exception as e:
            errors.append(f"Raw PCM: {e}")
        
        raise ValueError(f"Could not decode audio. Try using Chrome browser. Details: {'; '.join(errors[:2])}")

    def get_audio_duration(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> float:
        """Get duration of audio in seconds."""
        sr = sample_rate or self.sample_rate
        return len(audio) / sr

    def analyze_audio(self, audio: np.ndarray) -> AudioAnalysis:
        """Perform comprehensive audio analysis."""
        duration = self.get_audio_duration(audio)
        audio_normalized = self.normalize_audio(audio)
        
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        rms = librosa.feature.rms(y=audio_normalized, frame_length=frame_length, hop_length=hop_length)[0]
        avg_energy = np.mean(rms)
        
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_normalized, hop_length=hop_length)[0]
        avg_spectral_flatness = np.mean(spectral_flatness)
        
        speech_segments = self._detect_speech_segments(audio_normalized, rms, hop_length)
        has_speech = len(speech_segments) > 0
        
        is_whisper, whisper_confidence = self._detect_whisper(audio_normalized, avg_energy, avg_spectral_flatness)
        has_overlap, overlap_confidence = self._detect_overlapping_speech(audio_normalized, hop_length)
        
        return AudioAnalysis(
            audio=audio_normalized, sample_rate=self.sample_rate, duration=duration,
            has_speech=has_speech, is_whisper=is_whisper, whisper_confidence=whisper_confidence,
            has_overlapping_speech=has_overlap, overlap_confidence=overlap_confidence,
            speech_segments=speech_segments, avg_energy=float(avg_energy),
            spectral_flatness=float(avg_spectral_flatness)
        )

    def _detect_speech_segments(self, audio: np.ndarray, rms: np.ndarray, hop_length: int) -> List[Tuple[float, float]]:
        """Detect speech segments based on energy levels."""
        threshold = np.percentile(rms, 30)
        speech_frames = rms > threshold
        
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = start_frame * hop_length / self.sample_rate
                end_time = i * hop_length / self.sample_rate
                if end_time - start_time >= 0.1:
                    segments.append((start_time, end_time))
                in_speech = False
        
        if in_speech:
            start_time = start_frame * hop_length / self.sample_rate
            end_time = len(speech_frames) * hop_length / self.sample_rate
            if end_time - start_time >= 0.1:
                segments.append((start_time, end_time))
        
        return segments

    def _detect_whisper(self, audio: np.ndarray, avg_energy: float, avg_spectral_flatness: float) -> Tuple[bool, float]:
        """Detect whispered speech - stricter detection."""
        if avg_energy < 0.005:
            return False, 0.0
        
        confidence = 0.0
        
        # Low energy check
        if avg_energy < self.whisper_energy_threshold:
            energy_score = 1.0 - (avg_energy / self.whisper_energy_threshold)
            confidence += energy_score * 0.4
        
        # High spectral flatness
        if avg_spectral_flatness > self.whisper_spectral_flatness_threshold:
            flatness_score = min(1.0, (avg_spectral_flatness - 0.4) / 0.3)
            confidence += flatness_score * 0.3
        
        # Harmonic analysis
        try:
            harmonic, _ = librosa.effects.hpss(audio)
            harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(audio)) + 1e-6)
            if harmonic_ratio < 0.4:
                confidence += (1.0 - harmonic_ratio * 2.5) * 0.3
        except:
            pass
        
        is_whisper = confidence > 0.55  # Stricter threshold
        if is_whisper:
            logger.warning(f"Whisper detected! Confidence: {confidence:.2f}")
        
        return is_whisper, confidence

    def _detect_overlapping_speech(self, audio: np.ndarray, hop_length: int) -> Tuple[bool, float]:
        """
        Detect multiple speakers - FIXED with stricter detection.
        Only flags when there's strong evidence of multiple distinct voices.
        """
        try:
            confidence = 0.0
            
            # Method 1: Multi-pitch detection (primary method)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, hop_length=hop_length)
            
            multi_pitch_frames = 0
            total_voiced_frames = 0
            
            for frame_idx in range(pitches.shape[1]):
                frame_pitches = pitches[:, frame_idx]
                frame_mags = magnitudes[:, frame_idx]
                
                # Get significant pitches
                max_mag = np.max(frame_mags)
                if max_mag < 0.01:  # Skip quiet frames
                    continue
                    
                significant = frame_mags > max_mag * 0.5  # Stricter: 50% of max
                significant_pitches = frame_pitches[significant]
                significant_pitches = significant_pitches[(significant_pitches > 80) & (significant_pitches < 500)]  # Human voice range
                
                if len(significant_pitches) > 0:
                    total_voiced_frames += 1
                    
                    # Check for multiple distinct pitches
                    if len(significant_pitches) >= 2:
                        sorted_pitches = np.sort(significant_pitches)
                        # Check if any pair is NOT a harmonic relationship
                        found_non_harmonic = False
                        for i in range(len(sorted_pitches) - 1):
                            ratio = sorted_pitches[i + 1] / sorted_pitches[i]
                            # More strict harmonic check - ratios 1.5, 2, 2.5, 3, 4 are harmonics
                            harmonic_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
                            is_harmonic = any(abs(ratio - h) < 0.12 for h in harmonic_ratios)
                            if not is_harmonic and ratio > 1.2:  # Different fundamental
                                found_non_harmonic = True
                                break
                        if found_non_harmonic:
                            multi_pitch_frames += 1
            
            if total_voiced_frames > 10:  # Need enough voiced frames
                pitch_overlap_ratio = multi_pitch_frames / total_voiced_frames
                logger.debug(f"Overlap analysis: {multi_pitch_frames}/{total_voiced_frames} = {pitch_overlap_ratio:.2%}")
                
                # Only add confidence if significant portion has multiple pitches
                if pitch_overlap_ratio > self.overlap_pitch_threshold:
                    confidence += min(1.0, (pitch_overlap_ratio - 0.2) * 2) * 0.6
            
            # Method 2: Spectral complexity (secondary, lower weight)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            centroid_std = np.std(spectral_centroids)
            centroid_mean = np.mean(spectral_centroids)
            
            if centroid_mean > 0:
                cv = centroid_std / centroid_mean  # Coefficient of variation
                if cv > 0.4:  # High variation might indicate multiple speakers
                    confidence += min(1.0, (cv - 0.3) * 2) * 0.2
            
            # Method 3: Energy envelope analysis
            rms = librosa.feature.rms(y=audio)[0]
            if len(rms) > 10:
                rms_diff = np.abs(np.diff(rms))
                rapid_changes = np.sum(rms_diff > np.mean(rms) * 0.5)  # Stricter
                change_ratio = rapid_changes / len(rms_diff)
                if change_ratio > 0.25:  # Stricter threshold
                    confidence += min(1.0, (change_ratio - 0.2) * 3) * 0.2
            
            is_overlap = confidence > self.overlap_confidence_threshold
            
            if is_overlap:
                logger.warning(f"Overlapping speech detected! Confidence: {confidence:.2f}")
            else:
                logger.debug(f"No overlap detected. Confidence: {confidence:.2f}")
            
            return is_overlap, confidence
            
        except Exception as e:
            logger.warning(f"Overlap detection failed: {e}")
            return False, 0.0

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def extract_speech_segments(self, audio: np.ndarray) -> np.ndarray:
        """Extract speech using energy-based VAD."""
        try:
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            threshold = np.percentile(rms, 30)
            speech_frames = rms > threshold
            
            if not np.any(speech_frames):
                return audio
            
            speech_samples = []
            for i, is_speech in enumerate(speech_frames):
                if is_speech:
                    start_sample = i * hop_length
                    end_sample = min(start_sample + frame_length, len(audio))
                    speech_samples.append(audio[start_sample:end_sample])
            
            if not speech_samples:
                return audio
            
            speech_audio = np.concatenate(speech_samples)
            if self.get_audio_duration(speech_audio) < MIN_SPEECH_DURATION:
                return audio
            return speech_audio
        except Exception as e:
            logger.warning(f"VAD failed: {e}")
            return audio

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline."""
        audio = self.normalize_audio(audio)
        audio = self.extract_speech_segments(audio)
        return audio


_audio_processor: Optional[AudioProcessor] = None

def get_audio_processor() -> AudioProcessor:
    """Get or create the audio processor singleton."""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
