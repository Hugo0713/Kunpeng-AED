"""
Real-time audio acquisition module with rolling buffer support.
Supports both live microphone input and WAV file playback.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioStream:
    """Audio stream manager with producer-consumer pattern."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        window_duration: float = 0.96,
        hop_duration: float = 0.48,
        device: Optional[int] = None,
        wav_file: Optional[Path] = None
    ):
        """
        Initialize audio stream.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            channels: Number of audio channels (1=mono)
            window_duration: Rolling buffer window size in seconds
            hop_duration: Hop size between consecutive windows in seconds
            device: Audio device index (None=default)
            wav_file: Path to WAV file for playback mode
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_size = int(window_duration * sample_rate)
        self.hop_size = int(hop_duration * sample_rate)
        self.device = device
        self.wav_file = wav_file
        
        # Rolling buffer
        self.buffer = np.zeros(self.window_size, dtype=np.float32)
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info(
            f"AudioStream initialized: SR={sample_rate}Hz, "
            f"window={window_duration}s, hop={hop_duration}s"
        )
    
    def start(self) -> None:
        """Start audio capture thread."""
        if self.is_running:
            logger.warning("Stream already running")
            return
        
        self.is_running = True
        if self.wav_file:
            self._thread = threading.Thread(target=self._wav_worker, daemon=True)
        else:
            self._thread = threading.Thread(target=self._mic_worker, daemon=True)
        
        self._thread.start()
        logger.info("Audio stream started")
    
    def stop(self) -> None:
        """Stop audio capture thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Audio stream stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio frame from queue.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Audio frame of shape [window_size] or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _mic_worker(self) -> None:
        """Worker thread for live microphone capture."""
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            audio = indata[:, 0].astype(np.float32)
            self._update_buffer(audio)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.hop_size,
                device=self.device,
                callback=callback
            ):
                logger.info(f"Microphone opened (device={self.device})")
                while self.is_running:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Microphone error: {e}")
            self.is_running = False
    
    def _wav_worker(self) -> None:
        """Worker thread for WAV file playback."""
        try:
            audio_data, sr = sf.read(str(self.wav_file), dtype='float32')
            if sr != self.sample_rate:
                logger.warning(f"WAV sample rate {sr} != {self.sample_rate}")
            
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]  # Convert to mono
            
            logger.info(f"WAV file loaded: {len(audio_data)} samples")
            
            idx = 0
            while self.is_running and idx < len(audio_data):
                chunk = audio_data[idx:idx + self.hop_size]
                if len(chunk) < self.hop_size:
                    chunk = np.pad(chunk, (0, self.hop_size - len(chunk)))
                
                self._update_buffer(chunk)
                idx += self.hop_size
                threading.Event().wait(self.hop_duration)  # Simulate real-time
            
            logger.info("WAV playback finished")
        except Exception as e:
            logger.error(f"WAV playback error: {e}")
            self.is_running = False
    
    def _update_buffer(self, new_data: np.ndarray) -> None:
        """Update rolling buffer and emit new frame."""
        # Shift buffer left by hop_size
        self.buffer = np.roll(self.buffer, -self.hop_size)
        self.buffer[-self.hop_size:] = new_data[:self.hop_size]
        
        # Put frame in queue (non-blocking)
        try:
            self.frame_queue.put_nowait(self.buffer.copy())
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")


def test_stream():
    """Test audio stream with dummy playback."""
    logging.basicConfig(level=logging.INFO)
    
    # Create test WAV file
    test_wav = Path("/tmp/test_audio.wav")
    if not test_wav.exists():
        # Generate 3s of 440Hz sine wave
        t = np.linspace(0, 3, 3 * 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        sf.write(str(test_wav), audio, 16000)
        logger.info(f"Created test WAV: {test_wav}")
    
    stream = AudioStream(wav_file=test_wav)
    stream.start()
    
    for i in range(5):
        frame = stream.get_frame()
        if frame is not None:
            logger.info(f"Frame {i}: shape={frame.shape}, rms={np.sqrt(np.mean(frame**2)):.4f}")
    
    stream.stop()


if __name__ == "__main__":
    test_stream()
