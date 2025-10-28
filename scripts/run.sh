#!/bin/bash
# One-click start script for Kunpeng-AED

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Kunpeng-AED Startup Script"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check model file
MODEL_PATH="${MODEL_PATH:-models/yamnet_int8.tflite}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Please download YAMNet INT8 model first"
fi

# Parse arguments
THREADS="${THREADS:-4}"
PORT="${PORT:-8080}"
DEVICE="${DEVICE:-}"
WAV_FILE="${WAV_FILE:-}"

# Build command
CMD="python3 -m app.server --model $MODEL_PATH --threads $THREADS --port $PORT"

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$WAV_FILE" ]; then
    CMD="$CMD --wav $WAV_FILE"
fi

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Threads: $THREADS"
echo "  Port: $PORT"
echo ""
echo "Starting server..."
echo "Dashboard: http://localhost:$PORT"
echo ""

exec $CMD
