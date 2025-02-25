#!/bin/bash

echo "Starting NBA Prediction System..."
echo "Press Ctrl+C to stop the server."
echo ""
echo "The server will:"
echo "- Fetch player stats daily at 4:00 AM"
echo "- Update team rosters daily at 4:30 AM"
echo "- Get upcoming games daily at 5:00 AM"
echo "- Process data daily at 6:00 AM"
echo "- Train models weekly on Mondays at 7:00 AM"
echo "- Fetch projections and update predictions every 3 hours"
echo "- Verify predictions daily at 8:00 AM"
echo ""
echo "View the dashboard at: /Users/lukesmac/Models/nba/dashboard.html"
echo "Logs are available at: /Users/lukesmac/Models/nba/logs/unified_stats_fetcher.log"
echo ""

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Run the scheduler
python3 /Users/lukesmac/Models/nba/automated_pipeline.py --schedule