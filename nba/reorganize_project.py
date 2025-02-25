#!/usr/bin/env python3
"""
NBA Prediction System Reorganization Script

This script reorganizes the NBA prediction project structure to be more professional
and maintainable without breaking imports or functionality.
"""

import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path('/Users/lukesmac/Models')
NBA_DIR = BASE_DIR / 'nba'
PRIZEPICKS_DIR = BASE_DIR / 'prizepicks'

# Target directory structure
TARGET_DIR = BASE_DIR / 'nba-prediction-system'

# Directory mapping
DIR_MAPPING = {
    'config': TARGET_DIR / 'config',
    'data': TARGET_DIR / 'data',
    'data/raw': TARGET_DIR / 'data/raw',
    'data/processed': TARGET_DIR / 'data/processed',
    'data/features': TARGET_DIR / 'data/features',
    'data/predictions': TARGET_DIR / 'data/predictions',
    'logs': TARGET_DIR / 'logs',
    'logs/api': TARGET_DIR / 'logs/api',
    'logs/processing': TARGET_DIR / 'logs/processing',
    'logs/models': TARGET_DIR / 'logs/models',
    'models': TARGET_DIR / 'models',
    'scripts': TARGET_DIR / 'scripts',
    'scripts/fetch': TARGET_DIR / 'scripts/fetch',
    'scripts/process': TARGET_DIR / 'scripts/process',
    'scripts/projections': TARGET_DIR / 'scripts/projections',
    'utils': TARGET_DIR / 'utils',
    'dashboard': TARGET_DIR / 'dashboard',
    'tests': TARGET_DIR / 'tests',
}

# File mapping (source -> target)
FILE_MAPPING = {
    # NBA scripts
    NBA_DIR / 'scripts/getPlayerStats.py': TARGET_DIR / 'scripts/fetch/player_stats.py',
    NBA_DIR / 'scripts/getTeamRoster.py': TARGET_DIR / 'scripts/fetch/team_roster.py',
    NBA_DIR / 'scripts/getUpcomingGames.py': TARGET_DIR / 'scripts/fetch/upcoming_games.py',
    NBA_DIR / 'scripts/cleanStats.py': TARGET_DIR / 'scripts/process/clean_stats.py',
    NBA_DIR / 'scripts/update_unified_names.py': TARGET_DIR / 'scripts/process/update_unified_names.py',
    
    # NBA models
    NBA_DIR / 'models/fantasy_points.py': TARGET_DIR / 'models/fantasy_points.py',
    
    # Prizepicks scripts
    PRIZEPICKS_DIR / 'getProjections.py': TARGET_DIR / 'scripts/projections/fetch_projections.py',
    PRIZEPICKS_DIR / 'store_projections.py': TARGET_DIR / 'scripts/projections/store_projections.py',
    PRIZEPICKS_DIR / 'run_projections_scheduler.py': TARGET_DIR / 'scripts/projections/scheduler.py',
    
    # Logs - we'll just move the directories
    NBA_DIR / 'logs/game_stats_fetcher.log': TARGET_DIR / 'logs/api/game_stats_fetcher.log',
    NBA_DIR / 'logs/unified_stats_fetcher.log': TARGET_DIR / 'logs/api/unified_stats_fetcher.log',
    PRIZEPICKS_DIR / 'projections_scheduler.log': TARGET_DIR / 'logs/api/projections_scheduler.log',
}

# Create utility modules
UTILITY_FILES = {
    TARGET_DIR / 'utils/__init__.py': """
# NBA Prediction System Utilities
""",
    
    TARGET_DIR / 'utils/api.py': """
import requests
import logging
from config.api_config import NBA_API_KEY, NBA_API_HOST

def make_nba_api_request(endpoint, params=None):
    \"\"\"
    Make a request to the NBA API.
    
    Args:
        endpoint (str): API endpoint path
        params (dict): Query parameters
        
    Returns:
        dict: JSON response data or None if error
    \"\"\"
    url = f"https://{NBA_API_HOST}/{endpoint}"
    
    headers = {
        "x-rapidapi-key": NBA_API_KEY,
        "x-rapidapi-host": NBA_API_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request error: {e}")
        return None
""",
    
    TARGET_DIR / 'utils/db.py': """
import sqlite3
import pandas as pd
import logging
from contextlib import contextmanager
from config.paths import *

@contextmanager
def db_connection(db_path):
    \"\"\"Context manager for database connections\"\"\"
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        yield conn
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def query_to_df(db_path, query, params=None):
    \"\"\"Execute query and return results as DataFrame\"\"\"
    with db_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)

def save_df_to_db(df, db_path, table_name, if_exists='replace'):
    \"\"\"Save DataFrame to SQLite database\"\"\"
    with db_connection(db_path) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        logging.info(f"Saved {len(df)} records to {table_name} in {db_path}")
""",
    
    TARGET_DIR / 'utils/logging.py': """
import os
import logging
from datetime import datetime
from config.paths import LOGS_DIR

def setup_logger(name, category=None):
    \"\"\"Set up a logger with consistent formatting\"\"\"
    if category:
        log_dir = LOGS_DIR / category
    else:
        log_dir = LOGS_DIR
        
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
"""
}

# Configuration files
CONFIG_FILES = {
    TARGET_DIR / 'config/__init__.py': """
# NBA Prediction System Configuration
""",
    
    TARGET_DIR / 'config/paths.py': """
from pathlib import Path

# Base directories
BASE_DIR = Path('/Users/lukesmac/Models')
NBA_DIR = BASE_DIR / 'nba-prediction-system'
DATA_DIR = NBA_DIR / 'data'
LOGS_DIR = NBA_DIR / 'logs'
MODELS_DIR = NBA_DIR / 'models'
PROJECTIONS_DIR = NBA_DIR / 'data/projections'

# Database paths
PLAYER_STATS_DB = DATA_DIR / 'raw/player_game_stats.db'
CLEAN_STATS_DB = DATA_DIR / 'processed/clean_stats.db'
ENHANCED_STATS_DB = DATA_DIR / 'features/enhanced_stats.db'
PROJECTIONS_DB = PROJECTIONS_DIR / 'nba_projections.db'

# Create directories
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR, PROJECTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
""",
    
    TARGET_DIR / 'config/api_config.py': """
# API keys and configuration
NBA_API_KEY = "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade"
NBA_API_HOST = "tank01-fantasy-stats.p.rapidapi.com"
""",
}

# Pipeline file
PIPELINE_FILE = TARGET_DIR / 'pipeline.py'
PIPELINE_CONTENT = """
from pathlib import Path
import logging
import argparse
import os

from config.paths import *
from scripts.fetch import player_stats, team_roster, upcoming_games
from scripts.process import clean_stats
from models import features, predictions

def run_fetch_pipeline():
    \"\"\"Run data fetching pipeline\"\"\"
    player_stats.main()
    team_roster.main()
    upcoming_games.main()

def run_process_pipeline():
    \"\"\"Run data processing pipeline\"\"\"
    clean_stats.main()
    # features.enhance_features()
    
def run_model_pipeline():
    \"\"\"Run modeling pipeline\"\"\"
    # predictions.train_models()
    # predictions.generate_predictions()
    
def run_full_pipeline():
    \"\"\"Run complete pipeline\"\"\"
    run_fetch_pipeline()
    run_process_pipeline()
    run_model_pipeline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Prediction Pipeline")
    parser.add_argument("--step", choices=["fetch", "process", "model", "all"], 
                      default="all", help="Pipeline step to run")
    args = parser.parse_args()
    
    if args.step == "fetch":
        run_fetch_pipeline()
    elif args.step == "process":
        run_process_pipeline()
    elif args.step == "model":
        run_model_pipeline()
    else:
        run_full_pipeline()
"""

# Create __init__.py files for all directories
INIT_FILES = [
    TARGET_DIR / 'scripts/__init__.py',
    TARGET_DIR / 'scripts/fetch/__init__.py',
    TARGET_DIR / 'scripts/process/__init__.py',
    TARGET_DIR / 'scripts/projections/__init__.py',
    TARGET_DIR / 'models/__init__.py',
]

# README file
README_FILE = TARGET_DIR / 'README.md'
README_CONTENT = """# NBA Prediction System

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
"""

# Requirements file
REQUIREMENTS_FILE = TARGET_DIR / 'requirements.txt'
REQUIREMENTS_CONTENT = """
pandas>=2.0.0
numpy>=1.22.0
requests>=2.27.1
tqdm>=4.64.0
scikit-learn>=1.0.2
matplotlib>=3.5.0
seaborn>=0.11.2
"""

def create_directories():
    """Create the new directory structure"""
    for dir_path in DIR_MAPPING.values():
        logger.info(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)

def create_init_files():
    """Create __init__.py files for all directories"""
    for init_file in INIT_FILES:
        logger.info(f"Creating __init__.py file: {init_file}")
        with open(init_file, 'w') as f:
            f.write("# NBA Prediction System\n")

def create_utility_files():
    """Create utility modules"""
    for file_path, content in UTILITY_FILES.items():
        logger.info(f"Creating utility file: {file_path}")
        with open(file_path, 'w') as f:
            f.write(content)

def create_config_files():
    """Create configuration files"""
    for file_path, content in CONFIG_FILES.items():
        logger.info(f"Creating config file: {file_path}")
        with open(file_path, 'w') as f:
            f.write(content)

def create_pipeline_file():
    """Create main pipeline file"""
    logger.info(f"Creating pipeline file: {PIPELINE_FILE}")
    with open(PIPELINE_FILE, 'w') as f:
        f.write(PIPELINE_CONTENT)

def create_readme_file():
    """Create README file"""
    logger.info(f"Creating README file: {README_FILE}")
    with open(README_FILE, 'w') as f:
        f.write(README_CONTENT)

def create_requirements_file():
    """Create requirements.txt file"""
    logger.info(f"Creating requirements file: {REQUIREMENTS_FILE}")
    with open(REQUIREMENTS_FILE, 'w') as f:
        f.write(REQUIREMENTS_CONTENT)

def copy_files():
    """Copy files to new location with proper structure"""
    for source, target in FILE_MAPPING.items():
        if source.exists():
            logger.info(f"Copying {source} to {target}")
            # Create parent directories if they don't exist
            os.makedirs(target.parent, exist_ok=True)
            shutil.copy2(source, target)
        else:
            logger.warning(f"Source file not found: {source}")

def run_reorganization():
    """Run the full reorganization process"""
    logger.info("Starting project reorganization")
    
    # Create new structure
    create_directories()
    create_init_files()
    create_utility_files()
    create_config_files()
    create_pipeline_file()
    create_readme_file()
    create_requirements_file()
    
    # Copy files
    copy_files()
    
    logger.info(f"Project reorganization complete. New structure created at: {TARGET_DIR}")
    logger.info("NOTE: You will need to update imports in Python files to reference the new structure")
    logger.info("The original files have been preserved in their original locations")

if __name__ == "__main__":
    run_reorganization()