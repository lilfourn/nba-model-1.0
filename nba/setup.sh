#!/bin/bash

echo "Setting up NBA Prediction System..."

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p /Users/lukesmac/Models/nba/data
mkdir -p /Users/lukesmac/Models/nba/logs
mkdir -p /Users/lukesmac/Models/nba/dashboard_data
mkdir -p /Users/lukesmac/Models/prizepicks/data

# Install required packages
echo "Installing dependencies..."
pip install -r /Users/lukesmac/Models/nba/requirements.txt

# Make scripts executable
chmod +x /Users/lukesmac/Models/nba/automated_pipeline.py
chmod +x /Users/lukesmac/Models/nba/scripts/*.py
chmod +x /Users/lukesmac/Models/nba/models/*.py

echo "Installation complete! You can now run the system with:"
echo "python3 /Users/lukesmac/Models/nba/automated_pipeline.py --schedule"
echo ""
echo "Or to run a specific task:"
echo "python3 /Users/lukesmac/Models/nba/automated_pipeline.py --task fetch_player_stats"
echo ""
echo "To check the status of all tasks:"
echo "python3 /Users/lukesmac/Models/nba/automated_pipeline.py --status"