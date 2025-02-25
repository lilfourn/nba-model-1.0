
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
