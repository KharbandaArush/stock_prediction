#!/bin/bash

# Exit on error
set -e

echo "Starting installation and execution..."

# 1. System Dependencies
echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential

# 2. Python Environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install Python Packages
echo "Installing Python requirements..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

# 4. Run Pipeline
echo "Running stock prediction pipeline..."
# Ensure data directory exists
mkdir -p data
# Run the script
python run_pipeline.py

echo "Pipeline execution completed."
