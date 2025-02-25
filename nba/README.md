# NBA Predictions Automated Pipeline

This system automates the entire NBA prediction workflow, from data collection to model training, prediction generation, and performance tracking. The pipeline ensures predictions are always based on the latest data and continuously improves the model's accuracy by learning from historical performance.

## Features

- **Automated Data Collection**: Regularly fetches latest NBA player stats, team rosters, and upcoming games
- **Feature Engineering**: Creates enhanced features for improved prediction accuracy
- **Model Training**: Periodically retrains models with the latest data
- **Real-time Projections**: Compares our predictions with latest projection lines
- **Historical Tracking**: Records predictions and actual outcomes to track model performance
- **Dashboard Updates**: Automatically updates the visualization dashboard with fresh data
- **Fault Tolerance**: Robust error handling, retries, and logging
- **Scheduling**: Intelligent task scheduling to optimize resource usage

## Directory Structure

```
/Users/lukesmac/Models/
├── nba/
│   ├── data/                      # Data storage
│   │   ├── player_game_stats.db   # Raw player stats
│   │   ├── clean_stats.db         # Cleaned data
│   │   ├── enhanced_stats.db      # Data with additional features
│   │   ├── clean_stats_train.db   # Training data
│   │   ├── clean_stats_test.db    # Testing data
│   │   └── historical_predictions.db  # Prediction history and results
│   ├── models/                    # ML models
│   │   ├── trained/               # Trained model files
│   │   ├── player_predictions.py  # Main prediction logic
│   │   └── enhance_features.py    # Feature engineering
│   ├── scripts/                   # Data collection scripts
│   │   ├── getPlayerGameStats.py  # Get player game stats
│   │   ├── getTeamRoster.py       # Get team rosters
│   │   ├── getUpcomingGames.py    # Get scheduled games
│   │   └── cleanStats.py          # Data cleaning
│   ├── logs/                      # Log files
│   ├── dashboard_data/            # Data for visualization
│   ├── dashboard.html             # Dashboard UI
│   ├── automated_pipeline.py      # Main pipeline orchestrator
│   ├── nba-prediction-service.service  # systemd service file
│   └── com.lukesmac.nba-predictions.plist  # macOS launchd file
└── prizepicks/                    # Projections data
    ├── data/
    │   └── nba_projections.db     # Latest projection lines
    ├── getProjections.py          # Script to fetch projections
    └── store_projections.py       # Store projections
```

## Getting Started

### Prerequisites

- Python 3.8+ with the following packages:
  - pandas
  - numpy 
  - scikit-learn
  - xgboost
  - schedule
  - requests
  - sqlite3

### Installation

1. Clone or download the repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost schedule requests
   ```
3. Make sure the directory structure exists

### Running the Pipeline

#### Manual Execution

Run the complete pipeline once:
```bash
python3 /Users/lukesmac/Models/nba/automated_pipeline.py --pipeline
```

Run a specific task:
```bash
python3 /Users/lukesmac/Models/nba/automated_pipeline.py --task fetch_player_stats
```

Check pipeline status:
```bash
python3 /Users/lukesmac/Models/nba/automated_pipeline.py --status
```

Start the scheduler (runs continuously):
```bash
python3 /Users/lukesmac/Models/nba/automated_pipeline.py --schedule
```

#### Running as a System Service

##### On Linux (systemd)

1. Copy the service file to systemd directory:
   ```bash
   sudo cp /Users/lukesmac/Models/nba/nba-prediction-service.service /etc/systemd/system/
   ```

2. Reload systemd configuration:
   ```bash
   sudo systemctl daemon-reload
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl enable nba-prediction-service
   sudo systemctl start nba-prediction-service
   ```

4. Check status:
   ```bash
   sudo systemctl status nba-prediction-service
   ```

##### On macOS (launchd)

1. Copy the plist file to LaunchAgents directory:
   ```bash
   cp /Users/lukesmac/Models/nba/com.lukesmac.nba-predictions.plist ~/Library/LaunchAgents/
   ```

2. Load the service:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.lukesmac.nba-predictions.plist
   ```

3. Start the service:
   ```bash
   launchctl start com.lukesmac.nba-predictions
   ```

4. Check if it's running:
   ```bash
   launchctl list | grep nba-predictions
   ```

## Scheduled Tasks

The automated pipeline runs the following tasks on schedule:

- **Daily (4:00 AM)**: Fetch latest player game stats
- **Daily (4:30 AM)**: Update team rosters
- **Daily (5:00 AM)**: Get upcoming games
- **Daily (6:00 AM)**: Clean stats data
- **Daily (6:30 AM)**: Generate enhanced features
- **Weekly (Monday 7:00 AM)**: Retrain prediction models
- **Every 3 hours**: Fetch latest projections
- **Every 3 hours**: Generate fresh predictions
- **Every 3 hours**: Update dashboard data
- **Daily (8:00 AM)**: Verify prediction results
- **Weekly (Sunday 11:00 PM)**: Full pipeline refresh

## Dashboard

Access the prediction dashboard by opening:
```
/Users/lukesmac/Models/nba/dashboard.html
```

The dashboard shows:
- Latest predictions vs projections
- Confidence levels and explanations
- Model performance metrics
- Historical accuracy tracking
- Statistical visualizations

## Logs

Check the logs for detailed operation information:
```
/Users/lukesmac/Models/nba/logs/unified_stats_fetcher.log
```

For system service logs:
```
# Linux
journalctl -u nba-prediction-service

# macOS
cat /Users/lukesmac/Models/nba/logs/nba-predictions-output.log
cat /Users/lukesmac/Models/nba/logs/nba-predictions-error.log
```

## Historical Performance Tracking

The system automatically tracks the accuracy of predictions over time. This data is stored in:
```
/Users/lukesmac/Models/nba/data/historical_predictions.db
```

And visualized in the dashboard with metrics like:
- Overall hit rate
- Accuracy by confidence tier
- Accuracy by stat type
- Trend analysis

## Customizing the Pipeline

To modify task schedules or add new tasks:

1. Edit the `schedule_tasks()` method in `PipelineOrchestrator` class in `automated_pipeline.py`
2. Create new task classes by extending the `PipelineTask` base class
3. Add new tasks to the `self.tasks` dictionary in the `PipelineOrchestrator` constructor
4. Define dependencies in `self.task_dependencies`

## Troubleshooting

- **Pipeline fails**: Check the logs for detailed error messages
- **Scheduling issues**: Verify system time is correct
- **Data not updating**: Check API endpoints and database permissions
- **Model accuracy declining**: Review feature engineering and consider retraining

## Contributing

To contribute to this project:
1. Improve feature engineering in `enhance_features.py`
2. Enhance model performance in `player_predictions.py`
3. Add new data sources to enrich predictions
4. Optimize pipeline performance and resource usage