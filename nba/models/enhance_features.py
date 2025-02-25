import pandas as pd
import numpy as np
import sqlite3
import requests
import json
from datetime import datetime, timedelta
import logging
import os

# Configure logging
os.makedirs('/Users/lukesmac/Models/nba/logs', exist_ok=True)
logging.basicConfig(
    filename='/Users/lukesmac/Models/nba/logs/feature_engineering.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API configuration
API_KEY = "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade"
API_HOST = "tank01-fantasy-stats.p.rapidapi.com"

# Database paths
CLEAN_DB_PATH = '/Users/lukesmac/Models/nba/data/clean_stats.db'
ENHANCED_DB_PATH = '/Users/lukesmac/Models/nba/data/enhanced_stats.db'

def get_nba_games_for_date(game_date):
    """Get NBA games for a specific date from the Tank01 API."""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate"
    
    # Format date as YYYYMMDD
    if isinstance(game_date, datetime):
        formatted_date = game_date.strftime("%Y%m%d")
    else:
        formatted_date = game_date
    
    querystring = {"gameDate": formatted_date}
    
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching games for date {formatted_date}: {e}")
        return None

def get_nba_teams():
    """Get NBA teams data from the Tank01 API."""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeams"
    
    querystring = {
        "schedules": "true",
        "rosters": "true",
        "statsToGet": "averages",
        "topPerformers": "true",
        "teamStats": "true"
    }
    
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching NBA teams data: {e}")
        return None

def get_upcoming_week_games():
    """Get NBA games for the upcoming week."""
    today = datetime.now()
    games_by_date = {}
    
    # Get games for next 7 days
    for i in range(7):
        date = today + timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        logging.info(f"Fetching games for {date_str}...")
        games_data = get_nba_games_for_date(date_str)
        
        if games_data and "body" in games_data:
            games_by_date[date_str] = games_data["body"]
            game_count = len(games_data["body"]) if games_data["body"] else 0
            logging.info(f"Found {game_count} games for {date_str}")
        else:
            logging.warning(f"No games found or API error for {date_str}")
    
    return games_by_date

def extract_defensive_stats_by_position(teams_data):
    """Extract defensive stats by position for all teams."""
    defensive_stats = {}
    
    if teams_data and "body" in teams_data:
        for team_data in teams_data["body"]:
            team_abv = team_data.get('teamAbv')
            if not team_abv:
                continue
                
            # Get defensive stats
            team_def_stats = team_data.get('defensiveStats', {})
            if not team_def_stats:
                continue
                
            # Create a dictionary for this team
            defensive_stats[team_abv] = {
                'points_home': float(team_def_stats.get('ptsHome', 0)),
                'points_away': float(team_def_stats.get('ptsAway', 0)),
                'position_defense': {}
            }
            
            # Extract position-specific defensive stats
            for stat_type in ['blk', 'stl', 'fga', 'fgm', 'tptfgm', 'ast']:
                if stat_type not in team_def_stats:
                    continue
                    
                # Get stats by position
                for position in ['PG', 'SG', 'SF', 'PF', 'C']:
                    if position in team_def_stats[stat_type]:
                        # Initialize position data if not already there
                        if position not in defensive_stats[team_abv]['position_defense']:
                            defensive_stats[team_abv]['position_defense'][position] = {}
                        
                        # Add the specific stat
                        stat_value = float(team_def_stats[stat_type][position])
                        defensive_stats[team_abv]['position_defense'][position][stat_type] = stat_value
    
    return defensive_stats

def create_enhanced_features(clean_db_path, enhanced_db_path):
    """Create enhanced features from existing clean stats."""
    # Load data from clean_stats.db
    conn = sqlite3.connect(clean_db_path)
    query = "SELECT * FROM clean_player_stats"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert numeric columns stored as text back to float/int
    numeric_cols = [col for col in df.columns if col not in ['player_id', 'game_id', 'team', 'opponent', 'game_date', 'position', 'position_depth', 'name']]
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
    
    # Sort chronologically for time-series features
    df = df.sort_values(['player_id', 'game_date'])
    
    # Add last 5 games trend indicators (slope of recent performance)
    print("Calculating last 5 games trend indicators...")
    
    # Group by player
    for player_id, player_df in df.groupby('player_id'):
        if len(player_df) >= 5:
            player_idx = player_df.index
            
            # Calculate trend indicators for key stats using last 5 games
            for stat in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'fantasy_points']:
                # Get last 5 games stats
                last_5_games = player_df[stat].tail(5).values
                
                # Calculate slope using simple linear regression
                x = np.arange(len(last_5_games))
                slope, _ = np.polyfit(x, last_5_games, 1)
                
                # Normalize by the mean to get percentage trend per game
                mean_value = np.mean(last_5_games)
                if mean_value > 0:
                    normalized_slope = 100 * slope / mean_value
                else:
                    normalized_slope = 0
                
                # Add trend indicator
                df.loc[player_idx[-1], f'{stat}_trend_last_5'] = normalized_slope
    
    # Fill NAs in trend columns
    trend_cols = [col for col in df.columns if 'trend_last_5' in col]
    df[trend_cols] = df[trend_cols].fillna(0)
    
    # Add rest days features
    print("Adding rest days and schedule features...")
    
    # Calculate days rest and back-to-back indicators
    df['days_rest'] = df.groupby('player_id')['game_date'].transform(
        lambda x: x.astype('datetime64').diff().dt.days
    ).fillna(0)
    
    # Flag back-to-back games
    df['is_back_to_back'] = (df['days_rest'] <= 1).astype(int)
    
    # Flag long rest (4+ days)
    df['is_long_rest'] = (df['days_rest'] >= 4).astype(int)
    
    # Flag 3rd game in 4 nights
    df['game_date_dt'] = pd.to_datetime(df['game_date'])
    
    # For each player, check if this is 3rd game in 4 nights
    for player_id, player_df in df.groupby('player_id'):
        player_df = player_df.sort_values('game_date_dt')
        
        # Initialize 3rd game in 4 nights flag
        df.loc[df['player_id'] == player_id, 'is_3rd_game_in_4_nights'] = 0
        
        # Check for each game
        for i in range(2, len(player_df)):
            current_date = player_df.iloc[i]['game_date_dt']
            date_4_days_ago = current_date - timedelta(days=3)
            
            # Count games in last 4 days (including current game)
            games_in_4_days = sum(
                (player_df['game_date_dt'] >= date_4_days_ago) & 
                (player_df['game_date_dt'] <= current_date)
            )
            
            if games_in_4_days >= 3:
                idx = player_df.iloc[i].name  # Get the index
                df.loc[idx, 'is_3rd_game_in_4_nights'] = 1
    
    # Get team defensive stats
    print("Fetching team defensive stats...")
    
    # Get teams data
    teams_data = get_nba_teams()
    if teams_data:
        # Extract defensive stats by position
        defensive_stats = extract_defensive_stats_by_position(teams_data)
        
        # Add defensive stats columns
        df['opp_defense_vs_position_points'] = 0.0
        df['opp_defense_vs_position_rebounds'] = 0.0
        df['opp_defense_vs_position_assists'] = 0.0
        df['opp_defense_vs_position_blocks'] = 0.0
        df['opp_defense_vs_position_steals'] = 0.0
        
        # Fill with defensive stats for each player's position against their opponent
        for idx, row in df.iterrows():
            position = row['position']  # Player's position
            opponent = row['opponent']  # Opponent team
            
            # Check if we have defensive stats for this opponent and position
            if opponent in defensive_stats and position in defensive_stats[opponent]['position_defense']:
                pos_def = defensive_stats[opponent]['position_defense'][position]
                
                # Set defensive stats based on what's available
                df.loc[idx, 'opp_defense_vs_position_points'] = pos_def.get('fgm', 0) * 2 + pos_def.get('tptfgm', 0)
                df.loc[idx, 'opp_defense_vs_position_rebounds'] = 0.0  # Rebounding data not directly available
                df.loc[idx, 'opp_defense_vs_position_assists'] = pos_def.get('ast', 0)
                df.loc[idx, 'opp_defense_vs_position_blocks'] = pos_def.get('blk', 0)
                df.loc[idx, 'opp_defense_vs_position_steals'] = pos_def.get('stl', 0)
    
    # Save the enhanced data
    print(f"Saving enhanced data to {enhanced_db_path}...")
    conn = sqlite3.connect(enhanced_db_path)
    df.to_sql('enhanced_player_stats', conn, if_exists='replace', index=False)
    conn.close()
    
    print("Feature enhancement complete.")
    return df

if __name__ == "__main__":
    # Create enhanced features
    print("Creating enhanced features...")
    enhanced_df = create_enhanced_features(CLEAN_DB_PATH, ENHANCED_DB_PATH)
    print(f"Enhanced features created successfully. Total records: {len(enhanced_df)}")
    
    # Summarize new features
    new_features = [
        col for col in enhanced_df.columns 
        if any(x in col for x in ['trend_last_5', 'is_back_to_back', 
                                  'is_long_rest', 'is_3rd_game_in_4_nights',
                                  'opp_defense_vs_position'])
    ]
    
    print(f"\nNew features added ({len(new_features)}):")
    for feature in new_features:
        print(f"- {feature}")
    
    # Print sample of enhanced data
    print("\nSample of enhanced data:")
    sample_cols = ['name', 'game_date', 'position'] + new_features
    print(enhanced_df[sample_cols].sample(5).to_string())