import sys
import os
sys.path.append('/Users/lukesmac/Models/nba-prediction-system')
from utils.api import make_nba_api_request
from utils.db import db_connection, save_df_to_db
from utils.logging import setup_logger
from config.paths import DATA_DIR

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from statistics import mean

# Set up logging
logger = setup_logger("upcoming_games", "api")

# Database configuration
GAMES_DB_PATH = DATA_DIR / 'raw/upcoming_games.db'

def get_formatted_dates(num_days: int = 3) -> List[str]:
    """Get formatted dates for today and the next specified number of days."""
    dates = []
    current_date = datetime.now()
    
    for i in range(num_days):
        date = current_date + timedelta(days=i)
        formatted_date = date.strftime("%Y%m%d")
        dates.append(formatted_date)
    
    return dates

def calculate_average_odds(sportsbooks: List[Dict]) -> Dict[str, float]:
    """
    Calculate average moneyline odds from all sportsbooks.
    
    Args:
        sportsbooks: List of sportsbook data
        
    Returns:
        Dict with average home and away moneyline odds
    """
    home_odds = []
    away_odds = []
    
    for book in sportsbooks:
        if 'odds' in book:
            odds = book['odds']
            if 'homeTeamMLOdds' in odds and odds['homeTeamMLOdds']:
                try:
                    home_odds.append(float(odds['homeTeamMLOdds']))
                except ValueError:
                    pass
            if 'awayTeamMLOdds' in odds and odds['awayTeamMLOdds']:
                try:
                    away_odds.append(float(odds['awayTeamMLOdds']))
                except ValueError:
                    pass
    
    return {
        'avg_home_ml': round(mean(home_odds), 2) if home_odds else None,
        'avg_away_ml': round(mean(away_odds), 2) if away_odds else None
    }

def get_odds_for_date(date: str) -> Dict[str, Dict]:
    """
    Get betting odds for all games on a specific date.
    
    Args:
        date (str): Date in YYYYMMDD format
        
    Returns:
        Dict mapping game IDs to their odds
    """
    try:
        data = make_nba_api_request("getNBABettingOdds", {"gameDate": date, "itemFormat": "list"})
        
        odds_by_game = {}
        if data and 'body' in data:
            for game in data['body']:
                if 'gameID' in game and 'sportsBooks' in game:
                    odds_by_game[game['gameID']] = calculate_average_odds(game['sportsBooks'])
        
        return odds_by_game
    except Exception as e:
        logger.error(f"Error fetching odds for date {date}: {str(e)}")
        return {}

def get_games_for_date(date: str) -> Optional[Dict]:
    """Get NBA games scheduled for a specific date."""
    try:
        return make_nba_api_request("getNBAGamesForDate", {"gameDate": date})
    except Exception as e:
        logger.error(f"Error fetching games for date {date}: {str(e)}")
        return None

def get_upcoming_games() -> Dict[str, List[Dict]]:
    """
    Get NBA games and odds for today, tomorrow, and the day after.
    Only includes games that have odds data available.
    
    Returns:
        Dict[str, List[Dict]]: Dictionary mapping dates to lists of game data with odds
    """
    upcoming_games = {}
    dates = get_formatted_dates()
    
    for date in dates:
        games_data = get_games_for_date(date)
        odds_data = get_odds_for_date(date)
        
        if games_data and 'body' in games_data:
            games_list = []
            body_data = games_data['body']
            
            # Handle both dictionary and list responses
            if isinstance(body_data, dict):
                for game_id, game_info in body_data.items():
                    if isinstance(game_info, dict) and game_id in odds_data:
                        game_info['gameID'] = game_id
                        game_info.update(odds_data[game_id])
                        games_list.append(game_info)
            elif isinstance(body_data, list):
                for game in body_data:
                    if game['gameID'] in odds_data:
                        game.update(odds_data[game['gameID']])
                        games_list.append(game)
            
            upcoming_games[date] = games_list
            logger.info(f"Found {len(games_list)} games with odds for {date}")
        else:
            upcoming_games[date] = []
            logger.warning(f"No games found for {date}")
    
    return upcoming_games

def format_game_info(game: Dict) -> str:
    """Format game information into a readable string."""
    away_team = game.get('away', 'Unknown')
    home_team = game.get('home', 'Unknown')
    game_id = game.get('gameID', 'No ID')
    away_ml = game.get('avg_away_ml', 'No odds')
    home_ml = game.get('avg_home_ml', 'No odds')
    
    return f"{away_team} ({away_ml}) @ {home_team} ({home_ml}) - {game_id}"

def create_games_database():
    """Create SQLite database and tables for upcoming games."""
    with db_connection(GAMES_DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Create upcoming games table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS upcoming_games (
            date TEXT,
            game_id TEXT,
            home_team TEXT,
            away_team TEXT,
            start_time TEXT,
            home_ml_odds REAL,
            away_ml_odds REAL,
            game_data TEXT,
            PRIMARY KEY (date, game_id)
        )
        ''')
        
        conn.commit()
    
    logger.info("Games database and tables created successfully")

def store_upcoming_games(games: Dict[str, List[Dict]]):
    """
    Store upcoming games in SQLite database.
    
    Args:
        games: Dictionary of games by date
    """
    # Create database if it doesn't exist
    create_games_database()
    
    with db_connection(GAMES_DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM upcoming_games")
        
        # Insert data for each game
        for date, games_list in games.items():
            for game in games_list:
                home_team = game.get('home', '')
                away_team = game.get('away', '')
                game_id = game.get('gameID', '')
                start_time = game.get('startTime', '')
                home_ml = game.get('avg_home_ml')
                away_ml = game.get('avg_away_ml')
                
                # Store complete game data as JSON
                game_data = json.dumps(game)
                
                cursor.execute('''
                INSERT INTO upcoming_games 
                (date, game_id, home_team, away_team, start_time, home_ml_odds, away_ml_odds, game_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date, game_id, home_team, away_team, start_time, home_ml, away_ml, game_data
                ))
        
        conn.commit()
        logger.info(f"Stored upcoming games data for {len(games)} dates")

def main():
    """Main function to fetch and store upcoming NBA games and odds."""
    # Ensure data directory exists
    os.makedirs(DATA_DIR / 'raw', exist_ok=True)
    
    logger.info("Starting upcoming games fetching process")
    
    # Fetch games and odds
    games = get_upcoming_games()
    
    # Store in database
    store_upcoming_games(games)
    
    # Print summary
    for date, games_list in games.items():
        formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
        logger.info(f"Games for {formatted_date}: {len(games_list)}")
    
    logger.info("Upcoming games fetching process completed")

if __name__ == "__main__":
    main()