#!/bin/bash
# Stop all Kunpeng-AED processes

echo "Stopping Kunpeng-AED processes..."

pkill -f "python.*app.server" || echo "No running server found"

echo "All processes terminated"
