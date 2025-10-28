"""
Flask + SocketIO web server for real-time AED dashboard.
Streams inference results to web clients via WebSocket.
"""

import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import psutil
from flask import Flask, render_template
from flask_socketio import SocketIO

from app.feature import FeatureExtractor
from app.infer_tflite import YAMNetInference
from app.stream import AudioStream

logger = logging.getLogger(__name__)

# Global state
app = Flask(__name__, 
            template_folder='../web/templates',
            static_folder='../web/static')
app.config['SECRET_KEY'] = 'kunpeng-aed-secret-2025'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='gevent')

audio_stream: Optional[AudioStream] = None
feature_extractor: Optional[FeatureExtractor] = None
inference_engine: Optional[YAMNetInference] = None
processing_thread: Optional[threading.Thread] = None
is_running = False


def init_system(
    model_path: Path,
    num_threads: int = 4,
    device: Optional[int] = None,
    wav_file: Optional[Path] = None
):
    """Initialize all system components."""
    global audio_stream, feature_extractor, inference_engine
    
    logger.info("Initializing Kunpeng-AED system...")
    
    # Audio stream
    audio_stream = AudioStream(
        sample_rate=16000,
        channels=1,
        window_duration=0.96,
        hop_duration=0.48,
        device=device,
        wav_file=wav_file
    )
    
    # Feature extractor
    feature_extractor = FeatureExtractor(
        sample_rate=16000,
        n_fft=2048,
        hop_length=160,
        n_mels=64,
        num_workers=2
    )
    
    # Inference engine
    inference_engine = YAMNetInference(
        model_path=model_path,
        num_threads=num_threads,
        use_npu=False
    )
    
    logger.info("System initialized successfully")


def processing_loop():
    """Main processing loop: audio → features → inference → emit."""
    global is_running
    
    process = psutil.Process()
    frame_count = 0
    
    logger.info("Processing loop started")
    
    while is_running:
        try:
            # Get audio frame
            audio_frame = audio_stream.get_frame(timeout=1.0)
            if audio_frame is None:
                continue
            
            # Extract features
            features = feature_extractor.extract(audio_frame)
            
            # Run inference
            predictions, latency_ms = inference_engine.predict(features)
            
            # Get top-5 classes
            top_k = inference_engine.get_top_k(predictions, k=5)
            
            # Get CPU usage
            cpu_percent = process.cpu_percent(interval=None)
            
            # Prepare result
            result = {
                'timestamp': time.time(),
                'frame_id': frame_count,
                'top_class': top_k[0][0],
                'confidence': top_k[0][1],
                'top_k': [{'class': cls, 'prob': prob} for cls, prob in top_k],
                'latency_ms': latency_ms,
                'cpu_percent': cpu_percent,
                'threads': inference_engine.num_threads
            }
            
            # Emit to all connected clients
            socketio.emit('inference_result', result, namespace='/')
            
            frame_count += 1
            
            if frame_count % 50 == 0:
                logger.info(
                    f"Processed {frame_count} frames | "
                    f"Latency: {latency_ms:.2f}ms | "
                    f"CPU: {cpu_percent:.1f}%"
                )
                
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            time.sleep(0.1)
    
    logger.info("Processing loop stopped")


@app.route('/')
def index():
    """Serve main dashboard page."""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid if 'request' in dir() else 'unknown'}")
    
    # Send system status
    status = {
        'status': 'running' if is_running else 'stopped',
        'model': str(inference_engine.model_path) if inference_engine else 'N/A',
        'threads': inference_engine.num_threads if inference_engine else 0
    }
    socketio.emit('system_status', status)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")


def start_server(host='0.0.0.0', port=8080):
    """Start Flask-SocketIO server."""
    global is_running, processing_thread
    
    is_running = True
    
    # Start audio stream
    audio_stream.start()
    
    # Start processing thread
    processing_thread = threading.Thread(target=processing_loop, daemon=True)
    processing_thread.start()
    
    logger.info(f"Starting server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)


def stop_server():
    """Stop all components gracefully."""
    global is_running
    
    logger.info("Stopping Kunpeng-AED server...")
    
    is_running = False
    
    if audio_stream:
        audio_stream.stop()
    
    if feature_extractor:
        feature_extractor.close()
    
    if processing_thread:
        processing_thread.join(timeout=3.0)
    
    logger.info("Server stopped")


def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM."""
    logger.info(f"Received signal {sig}, shutting down...")
    stop_server()
    sys.exit(0)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kunpeng-AED Web Server')
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/yamnet_int8.tflite'),
        help='Path to TFLite model'
    )
    parser.add_argument('--threads', type=int, default=4, help='Inference threads')
    parser.add_argument('--device', type=int, default=None, help='Audio device index')
    parser.add_argument('--wav', type=Path, default=None, help='WAV file for playback')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize system
    init_system(
        model_path=args.model,
        num_threads=args.threads,
        device=args.device,
        wav_file=args.wav
    )
    
    # Start server
    start_server(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
