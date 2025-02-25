#!/bin/bash
# Script to generate data for the NBA Stats Prediction Dashboard

# Create dashboard data directory if it doesn't exist
DASHBOARD_DIR="/Users/lukesmac/Models/nba/dashboard_data"
mkdir -p "$DASHBOARD_DIR"

echo "Generating dashboard data..."

# Train models and generate dashboard data
python3 /Users/lukesmac/Models/nba/models/player_predictions.py \
  --dashboard-data "$DASHBOARD_DIR" \
  --compare-projections \
  --projections-db "/Users/lukesmac/Models/prizepicks/current_projections.db"

# Export projections comparison to CSV
python3 /Users/lukesmac/Models/nba/models/player_predictions.py \
  --compare-projections \
  --export-projections-csv "/Users/lukesmac/Models/nba/dashboard_data/latest_projections.csv"

echo "Dashboard data generated in $DASHBOARD_DIR"
echo "To view the dashboard, open the following file in your browser:"
echo "file:///Users/lukesmac/Models/nba/dashboard.html"