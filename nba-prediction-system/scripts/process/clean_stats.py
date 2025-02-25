import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import joblib
import os
import argparse
import logging
from pathlib import Path
import re

# Configure logging
os.makedirs('/Users/lukesmac/Models/nba/logs', exist_ok=True)
logging.basicConfig(
    filename='/Users/lukesmac/Models/nba/logs/clean_stats.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database paths
SOURCE_DB_PATH = '/Users/lukesmac/Models/nba/data/player_game_stats.db'
CLEAN_DB_PATH = '/Users/lukesmac/Models/nba/data/clean_stats.db'

def connect_to_db(db_path=SOURCE_DB_PATH):
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def load_data(db_path=SOURCE_DB_PATH):
    """Load data from SQLite database into a pandas DataFrame."""
    try:
        conn = connect_to_db(db_path)
        query = "SELECT * FROM player_game_stats"
        df = pd.read_sql_query(query, conn)
        logging.info(f"Loaded {len(df)} records from database")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    finally:
        if conn:
            conn.close()

def remove_duplicates(df):
    """Remove duplicate entries based on player, game, and date."""
    original_count = len(df)
    
    # Sort by date and keep the most recent entry in case of duplicates
    df = df.sort_values('game_date', ascending=False)
    df = df.drop_duplicates(subset=['player_id', 'game_id'], keep='first')
    
    removed_count = original_count - len(df)
    logging.info(f"Removed {removed_count} duplicate records")
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset with careful consideration of context.
    - Game absences (injuries, rest, DNPs) are represented as missing entries
    - Different missingness strategies are applied to different stat categories
    """
    original_df = df.copy()
    
    # Convert game_date to datetime to ensure proper sorting
    if 'game_date' in df.columns and not pd.api.types.is_datetime64_dtype(df['game_date']):
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Ensure data is sorted chronologically for each player
    df = df.sort_values(['player_id', 'game_date'])
    
    # Identify different types of statistics for custom handling
    game_stats = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 
                 'three_pointers_made', 'field_goals_made', 'field_goals_attempted',
                 'free_throws_made', 'free_throws_attempted']
    
    # Handle DNPs and absences appropriately
    # DNPs (Did Not Play) are represented by minutes = 0 and all stats = 0
    # This helps distinguish between missing data and intentional DNPs
    
    # First, detect likely DNPs where minutes are 0 or very low but game record exists
    if 'minutes' in df.columns:
        # Flag likely DNPs (player was on roster but didn't play meaningful minutes)
        df['is_dnp'] = ((df['minutes'].isna()) | (df['minutes'] < 1)) & (df['game_id'].notna())
        logging.info(f"Identified {df['is_dnp'].sum()} likely DNP records")
        
        # For DNPs, set game stats to 0 since player didn't play
        dnp_records = df['is_dnp'] == True
        for col in game_stats:
            if col in df.columns:
                missing_count = df.loc[dnp_records, col].isna().sum()
                if missing_count > 0:
                    df.loc[dnp_records, col] = 0
                    logging.info(f"Set {missing_count} DNP values in {col} to 0")
    
    # For regular missing values in statistical categories, use more nuanced approaches
    
    # 1. Fill missing game stats with 0 for statistical categories
    for col in game_stats:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(0)
                logging.info(f"Filled {missing_count} missing values in {col} with 0")
    
    # 2. Fill missing minutes with player-specific median for that season
    #    This ensures that minutes are filled with context-appropriate values
    if 'minutes' in df.columns and 'season' in df.columns:
        missing_minutes = df['minutes'].isna().sum()
        if missing_minutes > 0:
            # Create season-player groups for more accurate medians
            df['minutes'] = df.groupby(['player_id', 'season'])['minutes'].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
            )
            # Any remaining NaNs, fill with player's overall median
            df['minutes'] = df.groupby('player_id')['minutes'].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
            )
            # Remaining NaNs go to overall median
            df['minutes'] = df['minutes'].fillna(df['minutes'].median())
            logging.info(f"Filled {missing_minutes} missing minutes values with contextual medians")
    elif 'minutes' in df.columns:
        # If season not available, just use player medians
        missing_minutes = df['minutes'].isna().sum()
        if missing_minutes > 0:
            df['minutes'] = df.groupby('player_id')['minutes'].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
            )
            df['minutes'] = df['minutes'].fillna(df['minutes'].median())
            logging.info(f"Filled {missing_minutes} missing minutes values with player medians")
    
    # 3. Handle missing betting odds - these require special treatment
    #    Missing odds may indicate the game hasn't happened yet or odds weren't available
    ml_columns = [col for col in df.columns if 'ML_odds' in col or 'odds' in col.lower()]
    for col in ml_columns:
        missing_odds = df[col].isna().sum()
        if missing_odds > 0:
            # For historical games, fill with median by opponent to account for team strength
            if 'opponent' in df.columns:
                df[col] = df.groupby('opponent')[col].transform(
                    lambda x: x.fillna(x.median())
                )
            # Any remaining, use overall median
            df[col] = df[col].fillna(df[col].median())
            logging.info(f"Filled {missing_odds} missing values in {col} with contextual medians")
    
    # 4. Handle position data using the position and position_depth columns if available
    if 'position' in df.columns and 'position_depth' in df.columns:
        # Fill missing positions with most frequent position for that player
        missing_pos = df['position'].isna().sum()
        if missing_pos > 0:
            df['position'] = df.groupby('player_id')['position'].transform(
                lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
            )
            logging.info(f"Filled {missing_pos} missing position values with player's most common position")
        
        # Fill missing position_depth similarly
        missing_depth = df['position_depth'].isna().sum()
        if missing_depth > 0:
            df['position_depth'] = df.groupby(['player_id', 'position'])['position_depth'].transform(
                lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else x.name[0] + "?")
            )
            # Any remaining, try with just player_id
            df['position_depth'] = df.groupby('player_id')['position_depth'].transform(
                lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
            )
            logging.info(f"Filled {missing_depth} missing position_depth values")
    
    # 5. For non-numeric categorical columns, use forward-fill then backward-fill by player
    #    This maintains temporal consistency in categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'game_date' and col not in ['position', 'position_depth'] and col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df.groupby('player_id')[col].apply(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                # Any remaining NaNs fill with most common value
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
                logging.info(f"Filled {missing_count} missing values in {col} with player-specific patterns")
    
    # Clean up temporary columns
    if 'is_dnp' in df.columns:
        df = df.drop(columns=['is_dnp'])
    
    # Count total missing values before and after
    missing_before = original_df.isna().sum().sum()
    missing_after = df.isna().sum().sum()
    logging.info(f"Handled missing values: {missing_before} before, {missing_after} after")
    
    return df

def encode_categorical_features(df):
    """Encode categorical features into numeric format."""
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Binary encoding for home game indicator
    if 'home' in df.columns:
        # If it's not already numeric, convert it
        if df['home'].dtype != np.int64 and df['home'].dtype != np.bool_:
            df['home'] = df['home'].astype(int)
            logging.info("Encoded 'home' column to binary")
    
    # Binary encoding for win indicator
    if 'win' in df.columns:
        # If it's not already numeric, convert it
        if df['win'].dtype != np.int64 and df['win'].dtype != np.bool_:
            df['win'] = df['win'].astype(int)
            logging.info("Encoded 'win' column to binary")
    
    # Create encoders directory for model serialization
    encoders_dir = '/Users/lukesmac/Models/nba/models/encoders'
    os.makedirs(encoders_dir, exist_ok=True)
    
    # Encode team and opponent
    team_cols = ['team', 'opponent']
    for col in team_cols:
        if col in df.columns:
            le_team = LabelEncoder()
            df[f'{col}_encoded'] = le_team.fit_transform(df[col].astype(str))
            joblib.dump(le_team, f'{encoders_dir}/{col}_encoder.joblib')
            logging.info(f"Encoded '{col}' column with {len(le_team.classes_)} unique values")
    
    # Create additional date features
    if 'game_date' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df['game_date']):
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Extract useful date components
        df['game_year'] = df['game_date'].dt.year
        df['game_month'] = df['game_date'].dt.month
        df['game_dayofweek'] = df['game_date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = df['game_dayofweek'].isin([5, 6]).astype(int)  # Saturday or Sunday
        
        # NBA season typically starts in October (month 10)
        df['season'] = np.where(
            df['game_month'] >= 10,
            df['game_year'] + 1,  # e.g., games from Oct-Dec 2023 are season 2024
            df['game_year']       # e.g., games from Jan-Jun 2024 are still season 2024
        )
        
        # Days since start of season (useful for player fatigue/development)
        df['days_into_season'] = df.apply(
            lambda x: (x['game_date'] - pd.to_datetime(f"{int(x['season'])-1}-10-01")).days,
            axis=1
        )
        
        logging.info("Created date-related features from game_date")
    
    # Calculate days rest between games for each player
    df = df.sort_values(['player_id', 'game_date'])
    df['prev_game_date'] = df.groupby('player_id')['game_date'].shift(1)
    df['days_rest'] = (df['game_date'] - df['prev_game_date']).dt.days
    
    # Handle first game of season (no previous game)
    df['days_rest'] = df['days_rest'].fillna(5)  # Assume 5 days rest before first game
    logging.info("Created 'days_rest' feature")
    
    return df

def normalize_features(df):
    """Normalize features using appropriate scaling techniques."""
    # Create directory for scalers if it doesn't exist
    scaler_dir = '/Users/lukesmac/Models/nba/models/scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    
    # Create copy of dataframe
    df = df.copy()
    
    # Features to normalize with StandardScaler (for normal distributions)
    standard_scale_features = [
        'points', 'assists', 'rebounds', 'steals', 'blocks', 'turnovers',
        'three_pointers_made', 'field_goals_made', 'field_goals_attempted',
        'free_throws_made', 'free_throws_attempted', 'fantasy_points'
    ]
    
    # Add any rolling average features that might exist
    rolling_features = [col for col in df.columns if '_last_' in col]
    standard_scale_features.extend(rolling_features)
    
    # Features to normalize with MinMaxScaler (for bounded values)
    minmax_scale_features = [
        'minutes', 'days_rest', 'days_into_season',
        'home', 'win', 'is_weekend'
    ]
    
    # Features to skip normalization (keep as is)
    skip_features = [
        'id', 'player_id', 'game_id', 'game_date', 'team', 'opponent', 
        'prev_game_date', 'season'
    ]
    
    # Initialize scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    
    # Apply StandardScaler to appropriate features
    standard_features = [col for col in standard_scale_features if col in df.columns]
    if standard_features:
        # Check for NaN values
        for col in standard_features:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
                logging.warning(f"Found NaN values in {col} during normalization, filled with mean")
        
        # Apply scaler
        df[standard_features] = standard_scaler.fit_transform(df[standard_features])
        joblib.dump(standard_scaler, f'{scaler_dir}/standard_scaler.joblib')
        logging.info(f"Normalized {len(standard_features)} features with StandardScaler")
    
    # Apply MinMaxScaler to appropriate features
    minmax_features = [col for col in minmax_scale_features if col in df.columns]
    if minmax_features:
        # Check for NaN values
        for col in minmax_features:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
                logging.warning(f"Found NaN values in {col} during normalization, filled with median")
        
        # Apply scaler
        df[minmax_features] = minmax_scaler.fit_transform(df[minmax_features])
        joblib.dump(minmax_scaler, f'{scaler_dir}/minmax_scaler.joblib')
        logging.info(f"Normalized {len(minmax_features)} features with MinMaxScaler")
    
    # Special handling for odds-related features if they exist
    odds_features = [col for col in df.columns if 'ML_odds' in col]
    if odds_features:
        # Check if we have enough non-null values to scale
        if df[odds_features].notna().all(axis=1).any():
            odds_scaler = MinMaxScaler()
            # Fill any remaining NaNs with median
            for col in odds_features:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
            
            df[odds_features] = odds_scaler.fit_transform(df[odds_features])
            joblib.dump(odds_scaler, f'{scaler_dir}/odds_scaler.joblib')
            logging.info(f"Normalized {len(odds_features)} odds features")
    
    # Create a non-normalized version for reference
    df_non_normalized = df.copy()
    
    return df, df_non_normalized

def create_team_features(df):
    """Create team-related statistics and features."""
    logging.info("Creating team-related features")
    df = df.copy()
    
    # Only continue if we have team and opponent columns
    if 'team' in df.columns and 'opponent' in df.columns and 'game_date' in df.columns:
        # Calculate team offensive and defensive ratings
        df = df.sort_values(['game_date', 'team'])
        
        # Group by game date and team
        team_stats = df.groupby(['game_date', 'team']).agg({
            'points': 'mean',
            'rebounds': 'mean', 
            'assists': 'mean',
            'steals': 'mean',
            'blocks': 'mean',
            'turnovers': 'mean',
            'fantasy_points': 'mean'
        }).reset_index()
        
        # Rename columns to indicate these are team averages
        team_stats.columns = ['game_date', 'team', 'team_points_avg', 'team_rebounds_avg', 
                             'team_assists_avg', 'team_steals_avg', 'team_blocks_avg',
                             'team_turnovers_avg', 'team_fantasy_points_avg']
        
        # Merge team stats back to get team's performance that game
        df = pd.merge(
            df,
            team_stats,
            on=['game_date', 'team'],
            how='left'
        )
        
        # Also merge opponent stats
        opponent_stats = team_stats.rename(columns={
            'team': 'opponent',
            'team_points_avg': 'opp_points_avg',
            'team_rebounds_avg': 'opp_rebounds_avg',
            'team_assists_avg': 'opp_assists_avg',
            'team_steals_avg': 'opp_steals_avg',
            'team_blocks_avg': 'opp_blocks_avg',
            'team_turnovers_avg': 'opp_turnovers_avg',
            'team_fantasy_points_avg': 'opp_fantasy_points_avg'
        })
        
        df = pd.merge(
            df,
            opponent_stats,
            on=['game_date', 'opponent'],
            how='left'
        )
        
        # Calculate offensive and defensive ratings by team using rolling windows
        team_rolling_window = 10
        
        # Group data by team and sort by date for rolling calculations
        team_groups = []
        for team_name, team_data in df.sort_values('game_date').groupby('team'):
            # Calculate rolling offensive stats
            for stat in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'fantasy_points']:
                team_data[f'team_{stat}_rolling_{team_rolling_window}'] = team_data[f'team_{stat}_avg'].rolling(
                    window=team_rolling_window, min_periods=1).mean()
                
                # Also calculate the opponent's defensive stats against this team
                team_data[f'opp_allowed_{stat}_rolling_{team_rolling_window}'] = team_data[f'opp_{stat}_avg'].rolling(
                    window=team_rolling_window, min_periods=1).mean()
            
            team_groups.append(team_data)
        
        # Combine all team data back together
        df = pd.concat(team_groups)
        
        # Calculate team pace and efficiency metrics
        if 'possessions' not in df.columns and 'field_goals_attempted' in df.columns:
            # Estimate possessions (NBA formula approximation)
            df['team_possessions'] = df['team_points_avg'] + 0.4 * df['field_goals_attempted'] - 1.07 * (
                df['free_throws_attempted']) + 1.07 * df['team_turnovers_avg']
                
            df['opp_possessions'] = df['opp_points_avg'] + 0.4 * df['field_goals_attempted'] - 1.07 * (
                df['free_throws_attempted']) + 1.07 * df['opp_turnovers_avg']
            
            # Calculate offensive and defensive ratings (per 100 possessions)
            df['team_off_rating'] = 100 * df['team_points_avg'] / df['team_possessions']
            df['team_def_rating'] = 100 * df['opp_points_avg'] / df['opp_possessions']
            
            logging.info("Created team possession and rating metrics")
        
        logging.info(f"Created {df.shape[1] - team_stats.shape[1]} team-related features")
    else:
        logging.warning("Could not create team features - missing required columns")
    
    return df

def create_player_rolling_averages(df, windows=[3, 5, 10, 20]):
    """
    Create rolling average features for each player across multiple windows.
    
    Parameters:
    - df: DataFrame containing player game stats
    - windows: List of window sizes for rolling averages
    
    Returns:
    - DataFrame with rolling features added
    """
    logging.info(f"Creating player rolling averages with windows: {windows}")
    # Ensure dataframe is sorted chronologically
    df = df.sort_values(['player_id', 'game_date'])
    
    # Stats to create rolling averages for - expanded to include more metrics
    basic_stats = [
        'points', 'assists', 'rebounds', 'steals', 'blocks', 'turnovers',
        'minutes', 'fantasy_points', 'three_pointers_made', 'field_goals_made', 
        'field_goals_attempted', 'free_throws_made', 'free_throws_attempted'
    ]
    
    # Check if we have position data to create position-specific features
    has_position_data = 'position' in df.columns
    
    # Keep only columns that exist in the dataset
    stats_columns = [col for col in basic_stats if col in df.columns]
    
    # Store all player dataframes for later concatenation
    player_dfs = []
    
    # Process each player separately for more efficient processing
    player_ids = df['player_id'].unique()
    total_players = len(player_ids)
    
    # Track feature creation
    features_created = 0
    
    for i, player_id in enumerate(player_ids):
        if i % 50 == 0:  # Log progress every 50 players
            logging.info(f"Creating rolling features for player {i+1}/{total_players}")
            
        # Get player data - ensure chronological ordering
        player_df = df[df['player_id'] == player_id].sort_values('game_date').copy()
        
        # Create dictionaries to store new columns before adding them to the DataFrame
        new_columns = {}
        
        # 1. Create standard rolling averages for all window sizes
        for window in windows:
            for stat in stats_columns:
                # Calculate rolling averages (shift to avoid data leakage)
                col_name = f'{stat}_last_{window}'
                new_columns[col_name] = player_df[stat].shift(1).rolling(
                    window=window, min_periods=1).mean()
                features_created += 1
                
                # For key statistics, compute rolling std dev to capture consistency
                if stat in ['points', 'fantasy_points', 'minutes', 'rebounds', 'assists']:
                    # Standard deviation
                    col_name = f'{stat}_std_{window}'
                    new_columns[col_name] = player_df[stat].shift(1).rolling(
                        window=window, min_periods=max(3, window//2)).std()
                    features_created += 1
                    
                    # Min and max to capture range
                    col_name = f'{stat}_min_{window}'
                    new_columns[col_name] = player_df[stat].shift(1).rolling(
                        window=window, min_periods=max(3, window//2)).min()
                    features_created += 1
                    
                    col_name = f'{stat}_max_{window}'
                    new_columns[col_name] = player_df[stat].shift(1).rolling(
                        window=window, min_periods=max(3, window//2)).max()
                    features_created += 1
        
        # 2. Calculate shooting efficiency metrics
        if all(col in player_df.columns for col in ['field_goals_made', 'field_goals_attempted']):
            # Field goal percentage
            fg_pct = player_df['field_goals_made'] / player_df['field_goals_attempted']
            new_columns['fg_pct'] = fg_pct.replace([np.inf, -np.inf], np.nan).fillna(0)
            features_created += 1
            
            # Rolling FG% for each window
            for window in windows:
                col_name = f'fg_pct_last_{window}'
                new_columns[col_name] = new_columns['fg_pct'].shift(1).rolling(
                    window=window, min_periods=1).mean()
                features_created += 1
        
        if all(col in player_df.columns for col in ['three_pointers_made', 'field_goals_attempted']):
            # Three point attempt rate
            three_pt_rate = player_df['three_pointers_made'] / player_df['field_goals_attempted']
            new_columns['three_pt_rate'] = three_pt_rate.replace([np.inf, -np.inf], np.nan).fillna(0)
            features_created += 1
            
            # Rolling 3PT rate for each window
            for window in windows:
                col_name = f'three_pt_rate_last_{window}'
                new_columns[col_name] = new_columns['three_pt_rate'].shift(1).rolling(
                    window=window, min_periods=1).mean()
                features_created += 1
        
        if all(col in player_df.columns for col in ['free_throws_made', 'free_throws_attempted']):
            # Free throw percentage
            ft_pct = player_df['free_throws_made'] / player_df['free_throws_attempted']
            new_columns['ft_pct'] = ft_pct.replace([np.inf, -np.inf], np.nan).fillna(0)
            features_created += 1
            
            # Rolling FT% for each window
            for window in windows:
                col_name = f'ft_pct_last_{window}'
                new_columns[col_name] = new_columns['ft_pct'].shift(1).rolling(
                    window=window, min_periods=1).mean()
                features_created += 1
        
        # 3. Create context-specific features
        
        # Home vs away performance (if enough games)
        if len(player_df) >= 10 and 'home' in player_df.columns:
            # Calculate home/away splits for key statistics
            for stat in ['points', 'fantasy_points', 'rebounds', 'assists']:
                if stat in player_df.columns:
                    home_stat = player_df[player_df['home'] == 1][stat].mean()
                    away_stat = player_df[player_df['home'] == 0][stat].mean()
                    
                    # Home advantage (positive means better at home)
                    new_columns[f'home_advantage_{stat}'] = home_stat - away_stat
                    features_created += 1
        
        # Calculate trend indicators (recent form vs longer-term form)
        for stat in stats_columns:
            trend_cols = [f'{stat}_last_3', f'{stat}_last_10']
            if all(col in new_columns for col in trend_cols):
                new_columns[f'{stat}_trend'] = new_columns[trend_cols[0]] - new_columns[trend_cols[1]]
                features_created += 1
        
        # 4. Matchup-based features
        opponent_vs_cols = []
        if 'opponent' in player_df.columns:
            # Group by opponent to get average performance against each team
            opponent_stats = player_df.groupby('opponent')[stats_columns].mean().reset_index()
            # Rename columns to indicate these are vs specific opponent
            opponent_cols = [f'{col}_vs_opp_avg' for col in stats_columns]
            opponent_stats.columns = ['opponent'] + opponent_cols
            
            # Store the opponent columns for later merging
            for idx, row in opponent_stats.iterrows():
                opp_team = row['opponent']
                for col, val in zip(opponent_cols, row[1:]):
                    # Create a temporary series for each stat vs opponent
                    mask = player_df['opponent'] == opp_team
                    if mask.any():
                        if col not in new_columns:
                            new_columns[col] = pd.Series(index=player_df.index, dtype='float64')
                        new_columns[col].loc[mask] = val
                        
            opponent_vs_cols = opponent_cols
            features_created += len(stats_columns)
            
            # Performance vs opponent normalized by player's average (ratio)
            for stat in stats_columns:
                vs_col = f'{stat}_vs_opp_avg'
                if vs_col in opponent_vs_cols:
                    # Calculate player's overall average for this stat
                    player_avg = player_df[stat].mean()
                    if player_avg > 0:  # Avoid division by zero
                        # Calculate performance ratio
                        ratio_col = f'{stat}_vs_opp_ratio'
                        new_columns[ratio_col] = pd.Series(index=player_df.index, dtype='float64')
                        for idx in player_df.index:
                            if idx in new_columns[vs_col].index:
                                new_columns[ratio_col].loc[idx] = new_columns[vs_col].loc[idx] / player_avg
                        features_created += 1
        
        # 5. Position-specific features
        if has_position_data and 'position' in player_df.columns and not player_df['position'].isna().all():
            # Get player's primary position (most common)
            primary_position = player_df['position'].mode().iloc[0] if not player_df['position'].mode().empty else None
            
            if primary_position:
                # Set position flag for feature engineering
                new_columns['is_guard'] = player_df['position'].isin(['PG', 'SG']).astype(int)
                new_columns['is_forward'] = player_df['position'].isin(['SF', 'PF']).astype(int)
                new_columns['is_center'] = (player_df['position'] == 'C').astype(int)
                features_created += 3
                
                # Create position-specific metrics
                if primary_position in ['PG', 'SG']:  # Guards
                    # Assists-to-turnover ratio
                    if all(col in player_df.columns for col in ['assists', 'turnovers']):
                        ast_to = player_df['assists'] / player_df['turnovers'].replace(0, 0.5)
                        new_columns['ast_to_ratio'] = ast_to
                        features_created += 1
                        
                        # Rolling A/TO ratio
                        for window in [5, 10]:
                            col_name = f'ast_to_ratio_last_{window}'
                            new_columns[col_name] = new_columns['ast_to_ratio'].shift(1).rolling(
                                window=window, min_periods=1).mean()
                            features_created += 1
                
                elif primary_position in ['SF', 'PF']:  # Forwards
                    # Versatility score (points + rebounds + assists)
                    if all(col in player_df.columns for col in ['points', 'rebounds', 'assists']):
                        new_columns['versatility'] = player_df['points'] + player_df['rebounds'] + player_df['assists']
                        features_created += 1
                        
                        # Rolling versatility
                        for window in [5, 10]:
                            col_name = f'versatility_last_{window}'
                            new_columns[col_name] = new_columns['versatility'].shift(1).rolling(
                                window=window, min_periods=1).mean()
                            features_created += 1
                
                elif primary_position == 'C':  # Centers
                    # Rim protection (blocks + defensive rebounds)
                    if all(col in player_df.columns for col in ['blocks', 'rebounds']):
                        # Approximate defensive rebounds as 70% of total rebounds
                        new_columns['def_rebounds'] = player_df['rebounds'] * 0.7
                        new_columns['rim_protection'] = player_df['blocks'] + new_columns['def_rebounds']
                        features_created += 2
                        
                        # Rolling rim protection
                        for window in [5, 10]:
                            col_name = f'rim_protection_last_{window}'
                            new_columns[col_name] = new_columns['rim_protection'].shift(1).rolling(
                                window=window, min_periods=1).mean()
                            features_created += 1
        
        # 6. Game situation features
        
        # Check for back-to-back games
        days_since = (player_df['game_date'] - player_df['game_date'].shift(1)).dt.days
        new_columns['days_since_last_game'] = days_since
        new_columns['is_back_to_back'] = (days_since == 1).astype(int)
        new_columns['is_long_rest'] = (days_since >= 3).astype(int)
        features_created += 3
        
        # Performance on back-to-backs
        for stat in ['points', 'fantasy_points', 'minutes']:
            if stat in player_df.columns:
                b2b_mask = new_columns['is_back_to_back'] == 1
                rest_mask = new_columns['is_back_to_back'] == 0
                
                if b2b_mask.any() and rest_mask.any():
                    b2b_stat = player_df.loc[b2b_mask, stat].mean()
                    rest_stat = player_df.loc[rest_mask, stat].mean()
                    
                    # Impact of back-to-back on performance
                    new_columns[f'b2b_impact_{stat}'] = b2b_stat - rest_stat
                    features_created += 1
        
        # Win/loss performance differential (if available)
        if 'win' in player_df.columns:
            for stat in ['points', 'fantasy_points', 'minutes']:
                if stat in player_df.columns:
                    win_mask = player_df['win'] == 1
                    loss_mask = player_df['win'] == 0
                    
                    if win_mask.any() and loss_mask.any():
                        win_stat = player_df.loc[win_mask, stat].mean()
                        loss_stat = player_df.loc[loss_mask, stat].mean()
                        
                        # Performance differential in wins vs losses
                        new_columns[f'win_impact_{stat}'] = win_stat - loss_stat
                        features_created += 1
        
        # Add all new columns to the DataFrame at once to avoid fragmentation
        # Create a dictionary of all data
        all_data = {col: player_df[col] for col in player_df.columns}
        all_data.update(new_columns)
        # Create a new DataFrame with all columns at once
        player_df = pd.DataFrame(all_data, index=player_df.index)
        
        # Add to list of dataframes
        player_dfs.append(player_df)
    
    # Combine all player dataframes back together
    result_df = pd.concat(player_dfs)
    
    # Fill NAs with appropriate values
    for col in result_df.columns:
        # For rolling averages
        if any(col.endswith(f'_last_{w}') for w in windows) or any(col.endswith(f'_std_{w}') for w in windows) or \
           any(col.endswith(f'_min_{w}') for w in windows) or any(col.endswith(f'_max_{w}') for w in windows):
            if result_df[col].isna().any():
                # For rolling averages, fill with the overall average of that stat
                base_stat = re.match(r'(.+)_(last|std|min|max)_\d+', col)
                if base_stat and base_stat.group(1) in result_df.columns:
                    base_stat_name = base_stat.group(1)
                    result_df[col] = result_df[col].fillna(result_df[base_stat_name].mean())
                else:
                    result_df[col] = result_df[col].fillna(0)
        
        # For opponent-specific stats
        elif col.endswith('_vs_opp_avg') and result_df[col].isna().any():
            # Fill with player's overall average
            base_stat = col.split('_vs_opp_avg')[0]
            if base_stat in result_df.columns:
                result_df[col] = result_df.groupby('player_id')[col].transform(
                    lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else result_df[base_stat].mean())
                )
            else:
                result_df[col] = result_df[col].fillna(0)
        
        # For shooting percentages
        elif col in ['fg_pct', 'ft_pct', 'three_pt_rate'] or any(col.startswith(f'{x}_pct') for x in ['fg', 'ft']) or \
             col.startswith('three_pt_rate'):
            if result_df[col].isna().any():
                # Fill NaN percentages with overall mean
                result_df[col] = result_df[col].fillna(result_df[col].mean())
        
        # For other derived features
        elif result_df[col].isna().any() and col not in basic_stats and \
             not col.endswith('_id') and not col == 'game_date':
            result_df[col] = result_df[col].fillna(0)
    
    logging.info(f"Created {features_created} rolling and advanced features for {total_players} players")
    
    return result_df

def detect_outliers(df, columns=None, contamination=0.01):
    """
    Detect statistical outliers using Isolation Forest.
    
    Parameters:
    - df: DataFrame containing the data
    - columns: List of columns to check for outliers (defaults to numerical columns)
    - contamination: Expected proportion of outliers in the dataset
    
    Returns:
    - DataFrame with outlier flags and scores
    """
    df_copy = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        # Filter out ID columns and encoded categorical columns
        columns = [col for col in columns if not (col.endswith('_id') or 
                                                col.endswith('_encoded') or
                                                col == 'id')]
    
    # Ensure we have valid columns to work with
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logging.warning("No valid columns for outlier detection")
        return df_copy  # Return original with no changes
    
    # Prepare data for outlier detection (handling NaN values)
    X = df[valid_columns].copy()
    
    # Replace any remaining NaNs with column means
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())
    
    # Apply outlier detection algorithm
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_copy['outlier_score'] = iso_forest.fit_predict(X)
    
    # Convert to boolean flag (outlier = True)
    df_copy['is_outlier'] = df_copy['outlier_score'] == -1
    
    # Count outliers
    outlier_count = df_copy['is_outlier'].sum()
    logging.info(f"Detected {outlier_count} ({outlier_count/len(df_copy):.1%}) statistical outliers")
    
    return df_copy

def clean_obvious_errors(df, min_minutes_threshold=5, min_games_threshold=10):
    """
    Clean obvious errors and filter out low-quality records.
    Identifies, flags and handles outliers appropriately.
    """
    original_len = len(df)
    df = df.copy()
    
    # 1. Remove records with clearly impossible values
    
    # Impossible minutes values
    if 'minutes' in df.columns:
        invalid_minutes = df[df['minutes'] > 60].shape[0]
        if invalid_minutes > 0:
            df = df[df['minutes'] <= 60]
            logging.info(f"Removed {invalid_minutes} records with impossible minutes (> 60)")
    
    # Impossible statistical values
    stat_thresholds = {
        'points': 100,         # NBA record is 100 points (Wilt Chamberlain)
        'rebounds': 55,        # NBA record is 55 rebounds (Wilt Chamberlain)
        'assists': 30,         # NBA record is 30 assists (Scott Skiles)
        'steals': 11,          # NBA record is 11 steals (multiple players)
        'blocks': 17,          # NBA record is 17 blocks (Elmore Smith)
        'turnovers': 14,       # NBA record is 14 turnovers (Jason Kidd)
        'three_pointers_made': 14,  # NBA record is 14 threes (Klay Thompson)
        'field_goals_made': 36,     # NBA record is 36 FG (Wilt Chamberlain)
        'field_goals_attempted': 63, # NBA record is 63 FGA (Wilt Chamberlain)
        'free_throws_made': 28,     # NBA record is 28 FTs (Wilt Chamberlain)
        'free_throws_attempted': 39  # NBA record is 39 FTAs (Wilt Chamberlain)
    }
    
    for stat, threshold in stat_thresholds.items():
        if stat in df.columns:
            invalid_records = df[df[stat] > threshold].shape[0]
            if invalid_records > 0:
                df = df[df[stat] <= threshold]
                logging.info(f"Removed {invalid_records} records with impossible {stat} (> {threshold})")
    
    # 2. Detect statistical outliers (extreme but possible values)
    # These are flagged but not automatically removed
    df = detect_outliers(df)
    
    # Calculate career-high games (potential outliers but legitimate)
    if 'points' in df.columns and 'player_id' in df.columns:
        # For each player, calculate their career high
        df['career_high_points'] = df.groupby('player_id')['points'].transform('max')
        # Flag games that are within 5 points of career high
        df['is_career_high_game'] = (df['points'] >= df['career_high_points'] - 5)
        career_high_games = df['is_career_high_game'].sum()
        logging.info(f"Identified {career_high_games} potential career-high performances")
    
    # 3. Handle low minutes games
    # Low minutes can be intentional (blowouts, injuries) or data errors
    if 'minutes' in df.columns:
        low_minutes = df[df['minutes'] < min_minutes_threshold].shape[0]
        if low_minutes > 0:
            # Instead of removing, we'll flag these for later use in feature engineering
            df['low_minutes_game'] = df['minutes'] < min_minutes_threshold
            logging.info(f"Flagged {low_minutes} records with low minutes (< {min_minutes_threshold})")
            
            # For predictive modeling purposes, depending on the use case, we may filter these out
            filtered_df = df[df['minutes'] >= min_minutes_threshold].copy()
            logging.info(f"Filtered dataset has {len(filtered_df)} records after removing low-minutes games")
        else:
            filtered_df = df.copy()
    else:
        filtered_df = df.copy()
    
    # 4. Handle sample size issues (players with few games)
    player_game_counts = filtered_df.groupby('player_id').size()
    players_with_few_games = player_game_counts[player_game_counts < min_games_threshold].index
    
    if len(players_with_few_games) > 0:
        few_games_records = filtered_df[filtered_df['player_id'].isin(players_with_few_games)].shape[0]
        # Flag rather than remove
        filtered_df['few_games_player'] = filtered_df['player_id'].isin(players_with_few_games)
        logging.info(f"Flagged {few_games_records} records from {len(players_with_few_games)} players with <{min_games_threshold} games")
        
        # For statistical modeling, we may need to filter these out
        final_df = filtered_df[~filtered_df['player_id'].isin(players_with_few_games)].copy()
        logging.info(f"Final filtered dataset has {len(final_df)} records after all cleaning steps")
    else:
        final_df = filtered_df.copy()
    
    # Log statistics about the cleaning process
    total_removed = original_len - len(final_df)
    removal_pct = total_removed / original_len if original_len > 0 else 0
    logging.info(f"Cleaned {total_removed} ({removal_pct:.1%}) records with obvious errors or insufficient data")
    
    return final_df

def add_vegas_odds_features(df):
    """
    Create features based on Vegas odds and implied probabilities.
    
    Parameters:
    - df: DataFrame containing player stats with ML odds columns
    
    Returns:
    - DataFrame with additional odds-based features
    """
    df = df.copy()
    
    # Check if we have the necessary odds columns
    if all(col in df.columns for col in ['avg_home_ML_odds', 'avg_away_ML_odds']):
        logging.info("Creating Vegas odds-based features")
        
        # Convert American odds to implied probabilities
        # For positive odds: 100/(odds+100)
        # For negative odds: |odds|/(|odds|+100)
        
        # Home team implied win probability
        df['home_implied_prob'] = np.where(
            df['avg_home_ML_odds'] > 0,
            100 / (df['avg_home_ML_odds'] + 100),
            abs(df['avg_home_ML_odds']) / (abs(df['avg_home_ML_odds']) + 100)
        )
        
        # Away team implied win probability
        df['away_implied_prob'] = np.where(
            df['avg_away_ML_odds'] > 0,
            100 / (df['avg_away_ML_odds'] + 100),
            abs(df['avg_away_ML_odds']) / (abs(df['avg_away_ML_odds']) + 100)
        )
        
        # Team's implied win probability (based on home/away status)
        df['team_implied_prob'] = np.where(
            df['home'] == 1,
            df['home_implied_prob'],
            df['away_implied_prob']
        )
        
        # Opponent's implied win probability
        df['opponent_implied_prob'] = np.where(
            df['home'] == 1,
            df['away_implied_prob'],
            df['home_implied_prob']
        )
        
        # Calculate line value (favorite/underdog status)
        # Positive means team is favored, negative means team is underdog
        df['line_value'] = df['team_implied_prob'] - df['opponent_implied_prob']
        
        # Favorite/underdog status (1 = favorite, 0 = underdog)
        df['is_favorite'] = (df['line_value'] > 0).astype(int)
        
        # Create odds-based features
        # Performance vs expected (does player perform better when team is favored?)
        # Group by is_favorite for each player
        for stat in ['points', 'fantasy_points', 'minutes', 'rebounds', 'assists']:
            if stat in df.columns:
                # Calculate performance metrics by favorite status
                fav_group = df.groupby(['player_id', 'is_favorite'])[stat].mean().unstack()
                if 1 in fav_group.columns and 0 in fav_group.columns:
                    # Calculate performance differential (favorite vs underdog)
                    fav_diff = fav_group[1] - fav_group[0]
                    fav_diff_dict = fav_diff.to_dict()
                    # Add to dataframe
                    df[f'{stat}_fav_diff'] = df['player_id'].map(fav_diff_dict)
        
        logging.info(f"Created {7 + len([c for c in df.columns if c.endswith('_fav_diff')])} Vegas odds features")
    else:
        logging.warning("Could not create Vegas odds features - missing required ML odds columns")
    
    return df

def add_usage_metrics(df):
    """
    Create usage-based metrics to capture a player's role and involvement.
    
    Parameters:
    - df: DataFrame containing player stats
    
    Returns:
    - DataFrame with additional usage metrics
    """
    df = df.copy()
    
    # Calculate basic usage metrics
    if all(col in df.columns for col in ['field_goals_attempted', 'free_throws_attempted', 'turnovers']):
        logging.info("Creating usage metrics")
        
        # Usage rate approximation (simplified version of the actual formula)
        # We're missing possessions, but can approximate usage with available stats
        df['usage_rate'] = (df['field_goals_attempted'] + 0.44 * df['free_throws_attempted'] + df['turnovers']) / df['minutes']
        
        # Replace infinities and NaNs
        df['usage_rate'] = df['usage_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Shooting volume relative to minutes
        if 'field_goals_attempted' in df.columns:
            df['shot_volume'] = df['field_goals_attempted'] / df['minutes']
            df['shot_volume'] = df['shot_volume'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Create rolling usage metrics
        for window in [5, 10, 20]:
            # Group by player and sort by date
            df_sorted = df.sort_values(['player_id', 'game_date'])
            # Calculate rolling usage
            df[f'usage_rate_last_{window}'] = df_sorted.groupby('player_id')['usage_rate'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling shot volume if available
            if 'shot_volume' in df.columns:
                df[f'shot_volume_last_{window}'] = df_sorted.groupby('player_id')['shot_volume'].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
        
        logging.info("Created usage metrics with rolling windows")
    else:
        logging.warning("Could not create usage metrics - missing required columns")
    
    return df

def save_cleaned_data(df, output_path=CLEAN_DB_PATH):
    """
    Save the cleaned data to a new database file.
    Ensures data is properly sorted by date and indexed for efficient queries.
    
    Parameters:
    - df: DataFrame to save
    - output_path: Path to save the cleaned database
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Make a deep copy to avoid affecting the original dataframe
        df_to_save = df.copy(deep=True)
        
        # Handle potential Timestamp objects across all columns
        for col in df_to_save.columns:
            # First check for datetime dtypes
            if pd.api.types.is_datetime64_dtype(df_to_save[col]):
                df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d')
                logging.info(f"Converted datetime column {col} to string format")
                continue
                
            # Convert any columns with timestamps (not all might be identified as datetime)
            if df_to_save[col].dtype == 'object' or isinstance(df_to_save[col].dtype, pd.DatetimeTZDtype):
                # Sample some values (pandas Timestamp check is expensive for large columns)
                sample_size = min(100, len(df_to_save))
                sample_vals = df_to_save[col].dropna().head(sample_size)
                
                if len(sample_vals) > 0 and any(isinstance(x, (pd.Timestamp, datetime)) for x in sample_vals):
                    # Convert Timestamp objects to strings
                    df_to_save[col] = df_to_save[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp, datetime)) else x
                    )
                    logging.info(f"Converted Timestamp objects in column {col} to string format")
        
        # Sort data chronologically
        if 'game_date' in df_to_save.columns and 'player_id' in df_to_save.columns:
            df_to_save = df_to_save.sort_values(['player_id', 'game_date'])
        
        # Connect to the new database
        conn = sqlite3.connect(output_path)
        
        # Create a simple, clean schema with basic types
        logging.info(f"Preparing to save {len(df_to_save)} records with {len(df_to_save.columns)} columns")
        
        # Save to database - first create the table
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS clean_player_stats (
            {', '.join([f'"{col}" TEXT' for col in df_to_save.columns])}
        )
        """
        conn.execute("DROP TABLE IF EXISTS clean_player_stats")
        conn.execute(create_table_query)
        
        # Save in smaller chunks to avoid memory issues
        chunk_size = 5000
        for i in range(0, len(df_to_save), chunk_size):
            chunk = df_to_save.iloc[i:i+chunk_size]
            
            # Convert all data to strings to avoid SQLite type issues
            chunk_str = chunk.astype(str)
            
            # Insert data using executemany for better performance
            placeholders = ', '.join(['?'] * len(df_to_save.columns))
            insert_query = f'INSERT INTO clean_player_stats VALUES ({placeholders})'
            
            # Convert DataFrame to list of tuples for insertion
            data = [tuple(x) for x in chunk_str.values]
            conn.executemany(insert_query, data)
            
            conn.commit()
            logging.info(f"Saved chunk {i//chunk_size + 1}/{(len(df_to_save) + chunk_size - 1)//chunk_size}")
        
        # Create useful indices for efficient queries
        conn.execute('CREATE INDEX IF NOT EXISTS idx_player_id ON clean_player_stats (player_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_game_date ON clean_player_stats (game_date)')
        
        # Additional indices
        if 'position' in df_to_save.columns:
            conn.execute('CREATE INDEX IF NOT EXISTS idx_position ON clean_player_stats (position)')
        
        if 'team' in df_to_save.columns:
            conn.execute('CREATE INDEX IF NOT EXISTS idx_team ON clean_player_stats (team)')
        
        if 'opponent' in df_to_save.columns:
            conn.execute('CREATE INDEX IF NOT EXISTS idx_opponent ON clean_player_stats (opponent)')
        
        # Final commit
        conn.commit()
        
        # Log success
        logging.info(f"Successfully saved {len(df_to_save)} records to {output_path}")
        
        # Also save a CSV version for easier access (but only if not too large)
        if len(df_to_save) < 50000:  # Lower threshold for CSV export
            csv_path = output_path.replace('.db', '.csv')
            df_to_save.to_csv(csv_path, index=False)
            logging.info(f"Also saved data to CSV: {csv_path}")
        else:
            logging.info(f"Dataset too large ({len(df_to_save)} rows) for CSV export, skipping CSV creation")
        
    except Exception as e:
        logging.error(f"Error saving cleaned data: {e}")
        logging.error(f"Error details: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def create_train_test_split(df, test_ratio=0.2, split_date=None):
    """
    Split data into training and test sets.
    
    Parameters:
    - df: DataFrame containing player stats
    - test_ratio: Fraction of data to use for testing (if not using split_date)
    - split_date: If provided, splits data by date instead of ratio
    
    Returns:
    - train_df, test_df: Training and test DataFrames
    """
    if split_date:
        # Convert to datetime if it's a string
        if isinstance(split_date, str):
            split_date = pd.to_datetime(split_date)
        
        # Split based on date
        train_df = df[df['game_date'] < split_date].copy()
        test_df = df[df['game_date'] >= split_date].copy()
        logging.info(f"Split data by date ({split_date}): {len(train_df)} training samples, {len(test_df)} test samples")
    else:
        # Split the data chronologically for each player (most recent games as test)
        train_dfs = []
        test_dfs = []
        
        for player_id, player_df in df.groupby('player_id'):
            # Sort by date
            player_df = player_df.sort_values('game_date')
            
            # Calculate split point
            split_idx = int(len(player_df) * (1 - test_ratio))
            
            # Split data
            if len(player_df) <= 1:
                # If only one record, put it in train
                train_dfs.append(player_df)
            else:
                train_dfs.append(player_df.iloc[:split_idx])
                test_dfs.append(player_df.iloc[split_idx:])
        
        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame(columns=train_df.columns)
        
        logging.info(f"Split data by ratio ({1-test_ratio}:{test_ratio}): {len(train_df)} training samples, {len(test_df)} test samples")
    
    return train_df, test_df

def main():
    """Main function to clean NBA player game stats for ML preparation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clean NBA player game stats for ML preparation')
    parser.add_argument('--min-minutes', type=int, default=5, 
                      help='Minimum minutes played to include a game (default: 5)')
    parser.add_argument('--min-games', type=int, default=10, 
                      help='Minimum games played by a player to be included (default: 10)')
    parser.add_argument('--output-path', type=str, default=CLEAN_DB_PATH,
                      help=f'Path to save cleaned database (default: {CLEAN_DB_PATH})')
    parser.add_argument('--input-path', type=str, default=SOURCE_DB_PATH,
                      help=f'Path to source database (default: {SOURCE_DB_PATH})')
    parser.add_argument('--split-date', type=str, default=None,
                      help='Date to split train/test data (format: YYYY-MM-DD)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                      help='Ratio of data to use for testing (default: 0.2)')
    parser.add_argument('--keep-outliers', action='store_true',
                      help='Keep statistical outliers in the dataset')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize features (standardize/scale)')
    parser.add_argument('--detailed-features', action='store_true',
                      help='Create additional detailed features (slower but more comprehensive)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        print("Starting data cleaning process...")
        start_time = datetime.now()
        
        # 1. Load data
        print("Loading data from database...")
        df = load_data(args.input_path)
        original_len = len(df)
        
        # 2. Basic cleaning
        print("Removing duplicates...")
        df = remove_duplicates(df)
        
        print("Cleaning obvious errors and filtering low-quality records...")
        df = clean_obvious_errors(df, 
                               min_minutes_threshold=args.min_minutes,
                               min_games_threshold=args.min_games)
        
        # 3. Feature engineering
        print("Handling missing values...")
        df = handle_missing_values(df)
        
        print("Encoding categorical features...")
        df = encode_categorical_features(df)
        
        # 4. Advanced feature engineering
        print("Creating team-related features...")
        df = create_team_features(df)
        
        # Add player names if they exist
        if 'name' in df.columns:
            print("Player name column found, will preserve in clean dataset")
        
        # Create position-specific features
        if 'position' in df.columns:
            print("Position data found, will create position-specific features")
        
        # Create core rolling averages
        print("Creating player rolling averages...")
        df = create_player_rolling_averages(df, windows=[3, 5, 10, 20])
        
        # Add Vegas odds features if available
        if any('ML_odds' in col for col in df.columns):
            print("Creating Vegas odds features...")
            df = add_vegas_odds_features(df)
        
        # Add usage metrics
        print("Creating usage metrics...")
        df = add_usage_metrics(df)
        
        # 5. Create additional detailed features if requested
        if args.detailed_features:
            print("Creating additional detailed features...")
            # We could add more complex features here that are computationally expensive
            # For example, interaction terms, polynomial features, etc.
            
            # Create pairwise stat interactions for key metrics
            key_stats = ['points', 'rebounds', 'assists', 'minutes']
            available_key_stats = [stat for stat in key_stats if stat in df.columns]
            
            if len(available_key_stats) >= 2:
                print("Creating interaction features between key stats...")
                for i, stat1 in enumerate(available_key_stats):
                    for stat2 in available_key_stats[i+1:]:
                        df[f'{stat1}_{stat2}_interaction'] = df[stat1] * df[stat2]
            
            # More advanced position-specific metrics could be added here
            
            # Adjusted plus-minus or similar advanced metrics could be calculated here
        
        # 6. Final preparation
        if args.normalize:
            print("Normalizing features...")
            normalized_df, non_normalized_df = normalize_features(df)
        else:
            normalized_df = df
            non_normalized_df = df.copy()
            print("Skipping normalization as requested")
        
        # 7. Create train/test split
        print("Creating train/test split...")
        train_df, test_df = create_train_test_split(
            non_normalized_df, 
            test_ratio=args.test_ratio,
            split_date=args.split_date
        )
        
        # 8. Save all versions of the data
        print("Saving cleaned data...")
        # Save the main clean dataset
        save_cleaned_data(normalized_df, args.output_path)
        
        # Save train/test splits to separate files for ML pipelines
        train_path = args.output_path.replace('.db', '_train.db')
        test_path = args.output_path.replace('.db', '_test.db')
        
        # Save splits to SQLite
        save_cleaned_data(train_df, train_path)
        save_cleaned_data(test_df, test_path)
        logging.info(f"Saved train/test splits to {train_path} and {test_path}")
        
        # Calculate and log timing information
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log final statistics and timing
        print(f"\nData cleaning completed successfully in {duration:.1f} seconds!")
        print(f"Original records: {original_len}")
        print(f"Cleaned records: {len(normalized_df)}")
        print(f"Features created: {len(normalized_df.columns)}")
        print(f"Training samples: {len(train_df)} ({len(train_df)/len(normalized_df):.1%})")
        print(f"Test samples: {len(test_df)} ({len(test_df)/len(normalized_df):.1%})")
        print(f"\nCleaned data saved to: {args.output_path}")
        print(f"Training data saved to: {train_path}")
        print(f"Test data saved to: {test_path}")
        
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()