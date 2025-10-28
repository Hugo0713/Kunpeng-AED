"""
Audio feature extraction module using librosa.
Computes Mel-spectrogram with multi-core optimization.
"""

import logging
from multiprocessing import Pool
from typing import Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Mel-spectrogram feature extractor."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 160,
        n_mels: int = 64,
        fmin: float = 125.0,
        fmax: float = 7500.0,
        num_workers: int = 1
    ):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sampling rate
            n_fft: FFT window size
            hop_length: Hop size between frames
            n_mels: Number of Mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            num_workers: Number of parallel workers
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.num_workers = num_workers
        
        # Precompute Mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
        
        # Normalization statistics (placeholder, update with real stats)
        self.mean = 0.0
        self.std = 1.0
        
        self.pool: Optional[Pool] = None
        if num_workers > 1:
            self.pool = Pool(processes=num_workers)
        
        logger.info(
            f"FeatureExtractor initialized: n_mels={n_mels}, "
            f"n_fft={n_fft}, workers={num_workers}"
        )
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-Mel spectrogram from audio.
        
        Args:
            audio: Audio waveform of shape [samples]
            
        Returns:
            Log-Mel spectrogram of shape [n_mels, time_frames]
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        )
        
        # Power spectrogram
        power = np.abs(stft) ** 2
        
        # Apply Mel filterbank
        mel_spec = np.dot(self.mel_basis, power)
        
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - self.mean) / (self.std + 1e-6)
        
        return log_mel.astype(np.float32)
    
    def extract_batch(self, audio_batch: list[np.ndarray]) -> list[np.ndarray]:
        """
        Extract features from multiple audio clips in parallel.
        
        Args:
            audio_batch: List of audio arrays
            
        Returns:
            List of log-Mel spectrograms
        """
        if self.pool:
            return self.pool.map(self.extract, audio_batch)
        else:
            return [self.extract(audio) for audio in audio_batch]
    
    def close(self) -> None:
        """Close multiprocessing pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            logger.info("Feature extractor pool closed")
    
    def set_normalization_stats(self, mean: float, std: float) -> None:
        """Update normalization statistics."""
        self.mean = mean
        self.std = std
        logger.info(f"Normalization updated: mean={mean:.2f}, std={std:.2f}")


def test_feature_extraction():
    """Test feature extraction."""
    logging.basicConfig(level=logging.INFO)
    
    # Generate test audio (1s white noise)
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    extractor = FeatureExtractor(num_workers=2)
    features = extractor.extract(audio)
    
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
    
    # Batch test
    batch = [audio] * 4
    batch_features = extractor.extract_batch(batch)
    logger.info(f"Batch size: {len(batch_features)}")
    
    extractor.close()


if __name__ == "__main__":
    test_feature_extraction()
