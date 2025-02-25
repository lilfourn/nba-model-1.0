# NBA Prediction System

A machine learning system for predicting NBA player performance and comparing against projections.

## Project Structure

```
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── db_config.py         # Database configuration
│   ├── api_config.py        # API keys and endpoints
│   └── paths.py             # Central path definitions
├── data/                    # Data directory
│   ├── raw/                 # Raw data from APIs
│   ├── processed/           # Cleaned and processed data
│   ├── features/            # Feature engineered data
│   └── predictions/         # Model predictions
├── logs/                    # Log files
│   ├── api/                 # API-related logs
│   ├── processing/          # Data processing logs
│   └── models/              # Model training logs
├── models/                  # ML model code
│   ├── __init__.py
│   └── fantasy_points.py    # Fantasy points prediction model
├── scripts/                 # Data collection scripts
│   ├── __init__.py
│   ├── fetch/               # Data fetching scripts
│   │   ├── __init__.py
│   │   ├── player_stats.py  # Player stats fetcher
│   │   ├── team_roster.py   # Team roster fetcher
│   │   └── upcoming_games.py # Upcoming games fetcher
│   ├── process/             # Data processing scripts
│   │   ├── __init__.py
│   │   └── clean_stats.py   # Data cleaning
│   └── projections/         # Projection scripts
│       ├── __init__.py
│       ├── fetch_projections.py    # Fetch projections
│       └── store_projections.py    # Store projections
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── db.py                # Database utilities
│   ├── api.py               # API utilities
│   └── logging.py           # Logging utilities
├── pipeline.py              # Main pipeline script
└── README.md                # Project documentation
```

## Getting Started

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the full pipeline:
   ```
   python pipeline.py --step all
   ```

## Usage

### Data Collection
- Fetch player stats: `python scripts/fetch/player_stats.py`
- Fetch team rosters: `python scripts/fetch/team_roster.py`
- Fetch upcoming games: `python scripts/fetch/upcoming_games.py`

### Data Processing
- Clean stats: `python scripts/process/clean_stats.py`

### Projections
- Fetch projections: `python scripts/projections/fetch_projections.py`
- Store projections: `python scripts/projections/store_projections.py`
- Run projections scheduler: `python scripts/projections/scheduler.py`

### Pipeline
- Run complete pipeline: `python pipeline.py`
- Run specific step: `python pipeline.py --step [fetch|process|model]`
