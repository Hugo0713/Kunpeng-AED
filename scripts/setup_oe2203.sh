#!/bin/bash
# Setup script for openEuler 22.03 on Kunpeng platform

set -e

echo "=========================================="
echo "  Kunpeng-AED Environment Setup"
echo "  Target: openEuler 22.03 + Kunpeng"
echo "=========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo dnf update -y

# Install Python 3.10+
echo "[2/6] Installing Python 3.10..."
sudo dnf install -y python3 python3-pip python3-devel

# Install system dependencies
echo "[3/6] Installing system libraries..."
sudo dnf install -y \
    gcc gcc-c++ make cmake \
    portaudio-devel libsndfile-devel \
    alsa-lib-devel

# Create virtual environment
echo "[4/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python packages
echo "[5/6] Installing Python dependencies..."
pip install -r requirements.txt

# Download model (placeholder)
echo "[6/6] Downloading YAMNet INT8 model..."
mkdir -p models
echo "Please manually download yamnet_int8.tflite to models/"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo "Next steps:"
echo "  1. Download YAMNet model to models/"
echo "  2. Run: bash scripts/run.sh"
echo "  3. Open: http://<board-ip>:8080"
