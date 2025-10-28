"""
TFLite inference engine for YAMNet INT8 model.
Supports multi-threading and optional NPU delegation.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

logger = logging.getLogger(__name__)


class YAMNetInference:
    """YAMNet INT8 TFLite inference engine."""
    
    def __init__(
        self,
        model_path: Path,
        num_threads: int = 4,
        use_npu: bool = False,
        class_names: Optional[list[str]] = None
    ):
        """
        Initialize TFLite inference engine.
        
        Args:
            model_path: Path to YAMNet INT8 TFLite model
            num_threads: Number of CPU threads for inference
            use_npu: Enable NPU delegate (if available)
            class_names: List of 521 class names
        """
        self.model_path = model_path
        self.num_threads = num_threads
        self.use_npu = use_npu
        
        # Load model
        delegates = []
        if use_npu:
            try:
                # Placeholder for NPU delegate (platform-specific)
                logger.warning("NPU delegate not implemented, using CPU")
            except Exception as e:
                logger.warning(f"NPU delegate failed: {e}")
        
        self.interpreter = tflite.Interpreter(
            model_path=str(model_path),
            num_threads=num_threads,
            experimental_delegates=delegates
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        self.input_shape = self.input_details['shape']
        self.output_shape = self.output_details['shape']
        
        # Class names (YAMNet has 521 AudioSet classes)
        self.class_names = class_names or [f"class_{i}" for i in range(521)]
        
        logger.info(
            f"YAMNet loaded: input={self.input_shape}, "
            f"output={self.output_shape}, threads={num_threads}"
        )
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on input features.
        
        Args:
            features: Log-Mel spectrogram of shape [n_mels, time_frames]
                     Will be reshaped to [1, 96, 64] for YAMNet
            
        Returns:
            (predictions, latency_ms): Class probabilities and inference time
        """
        # Reshape features to model input shape [1, 96, 64]
        if features.shape[0] != 64:
            logger.warning(f"Expected 64 Mel bands, got {features.shape[0]}")
        
        # Crop or pad time dimension to 96 frames
        if features.shape[1] < 96:
            features = np.pad(features, ((0, 0), (0, 96 - features.shape[1])))
        elif features.shape[1] > 96:
            features = features[:, :96]
        
        # Transpose to [96, 64] and add batch dimension
        input_data = features.T[np.newaxis, ...].astype(np.float32)
        
        # Run inference
        start_time = time.perf_counter()
        
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details['index'])[0]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return predictions, latency_ms
    
    def get_top_k(
        self,
        predictions: np.ndarray,
        k: int = 5
    ) -> list[Tuple[str, float]]:
        """
        Get top-K predicted classes.
        
        Args:
            predictions: Class probability vector
            k: Number of top classes to return
            
        Returns:
            List of (class_name, probability) tuples
        """
        top_indices = np.argsort(predictions)[-k:][::-1]
        return [
            (self.class_names[idx], float(predictions[idx]))
            for idx in top_indices
        ]


def test_inference():
    """Test TFLite inference with dummy data."""
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy model path (replace with actual model)
    model_path = Path("/x:/VS code/2025 Autumn/kunpeng/models/yamnet_int8.tflite")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Initialize engine
    engine = YAMNetInference(model_path, num_threads=4)
    
    # Create dummy input (64 Mel bands, 100 time frames)
    dummy_features = np.random.randn(64, 100).astype(np.float32)
    
    # Run inference
    predictions, latency = engine.predict(dummy_features)
    logger.info(f"Inference latency: {latency:.2f} ms")
    
    # Get top-5 classes
    top_k = engine.get_top_k(predictions, k=5)
    for i, (class_name, prob) in enumerate(top_k):
        logger.info(f"  {i+1}. {class_name}: {prob:.4f}")


if __name__ == "__main__":
    test_inference()
