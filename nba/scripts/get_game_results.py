#!/usr/bin/env python3
"""
NBA Game Results Fetcher

This script fetches actual game results from the NBA API and verifies
prediction accuracy by comparing the actual player stats with our predictions.
It updates the historical predictions database with verified results.
"""

import os
import sys
import requests
import sqlite3
import logging
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Set up paths
BASE_DIR = Path('/Users/lukesmac/Models')
NBA_DIR = BASE_DIR / 'nba'
DATA_DIR = NBA_DIR / 'data'
LOGS_DIR = NBA_DIR / 'logs'
HISTORICAL_DB = DATA_DIR / 'historical_predictions.db'

# Set up logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOGS_DIR / 'game_results_fetcher.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# API details (for the NBA Stats API)
NBA_STATS_API_BASE = "https://stats.nba.com/stats"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.nba.com'
}

# API Key for Tank01 API
TANK01_API_KEY = "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade"
TANK01_API_HOST = "tank01-fantasy-stats.p.rapidapi.com"


def get_unverified_predictions(days_back=7):
    """
    Get unverified predictions from the historical database.
    
    Args:
        days_back: How many days back to look for unverified predictions
        
    Returns:
        DataFrame of unverified predictions
    """
    # Calculate the date range
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=days_back)
    
    with sqlite3.connect(HISTORICAL_DB) as conn:
        query = """
        SELECT id, player_name, team, opponent, stat_type, line_score, 
               our_prediction, prediction, prediction_date
        FROM historical_predictions
        WHERE verified_date IS NULL 
          AND prediction_date BETWEEN ? AND ?
        """
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        )
    
    logging.info(f"Found {len(df)} unverified predictions between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
    return df


def get_game_date_from_prediction(prediction_date, team, opponent):
    """
    Find the actual game date based on prediction date and teams.
    In most cases, the prediction is made for a game on the same day
    or the next day.
    
    Args:
        prediction_date: Date the prediction was made
        team: Player's team
        opponent: Opponent team
        
    Returns:
        Game date string in YYYY-MM-DD format or None if not found
    """
    # First, try the prediction date itself
    game_date = prediction_date
    
    # If that doesn't work, try the next day
    if not find_game_by_teams(game_date, team, opponent):
        game_date = (datetime.strptime(prediction_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # If still not found, try the day before
        if not find_game_by_teams(game_date, team, opponent):
            game_date = (datetime.strptime(prediction_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            
            if not find_game_by_teams(game_date, team, opponent):
                return None
    
    return game_date


def find_game_by_teams(game_date, team, opponent):
    """
    Check if a game between the specified teams exists on the given date.
    
    Args:
        game_date: Date to check in YYYY-MM-DD format
        team: First team
        opponent: Second team
        
    Returns:
        Boolean indicating if game was found
    """
    try:
        # Use Tank01 API to get games for the date
        url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate"
        
        # Format date as YYYYMMDD for the API
        formatted_date = game_date.replace('-', '')
        
        querystring = {"gameDate": formatted_date}
        
        headers = {
            "x-rapidapi-key": TANK01_API_KEY,
            "x-rapidapi-host": TANK01_API_HOST
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        
        # Check if any games match the teams
        if 'body' in data and data['body']:
            for game in data['body']:
                if (game.get('home') == team and game.get('away') == opponent) or \
                   (game.get('home') == opponent and game.get('away') == team):
                    return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error checking for game: {e}")
        return False


def get_player_stats_for_game(player_name, game_date, team=None):
    """
    Get a player's stats for a specific game.
    
    Args:
        player_name: Name of the player
        game_date: Date of the game in YYYY-MM-DD format
        team: Player's team (optional)
        
    Returns:
        Dictionary of player stats or None if not found
    """
    try:
        # Use Tank01 API to get player game stats
        url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerGameStats"
        
        # Format date as YYYYMMDD for the API
        formatted_date = game_date.replace('-', '')
        
        querystring = {
            "playerName": player_name,
            "gameDate": formatted_date,
            "teamABV": team
        }
        
        headers = {
            "x-rapidapi-key": TANK01_API_KEY,
            "x-rapidapi-host": TANK01_API_HOST
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        
        if 'body' in data and data['body']:
            return data['body']
        
        logging.warning(f"No stats found for {player_name} on {game_date}")
        return None
        
    except Exception as e:
        logging.error(f"Error getting player stats: {e}")
        return None


def get_stat_value_from_results(game_stats, stat_type):
    """
    Extract the relevant stat value from game stats based on stat type.
    
    Args:
        game_stats: Dictionary of game statistics
        stat_type: Type of stat to extract ('Points', 'Rebounds', etc.)
        
    Returns:
        Numeric value of the stat or None if not found
    """
    # Map from our stat_type to the API's field names
    stat_mapping = {
        'Points': 'pts',
        'Rebounds': 'reb',
        'Assists': 'ast',
        'Steals': 'stl',
        'Blocks': 'blk',
        'Turnovers': 'TOV',
        '3-PT Made': 'tptfgm',
        'FG Made': 'fgm',
        'Fantasy Score': None,  # Need to calculate this
        'Free Throws Made': 'ftm'
    }
    
    if stat_type not in stat_mapping:
        logging.error(f"Unknown stat type: {stat_type}")
        return None
    
    if stat_type == 'Fantasy Score':
        # Calculate fantasy score based on our scoring system
        try:
            pts = float(game_stats.get('pts', 0))
            reb = float(game_stats.get('reb', 0))
            ast = float(game_stats.get('ast', 0))
            stl = float(game_stats.get('stl', 0))
            blk = float(game_stats.get('blk', 0))
            tov = float(game_stats.get('TOV', 0))
            
            # Fantasy points formula from earlier code
            fantasy_score = (
                pts * 1.0 + 
                reb * 1.25 + 
                ast * 1.5 + 
                stl * 3.0 + 
                blk * 3.0 + 
                tov * -1.0
            )
            
            return fantasy_score
        except (ValueError, TypeError) as e:
            logging.error(f"Error calculating fantasy score: {e}")
            return None
    else:
        # Get the direct stat
        stat_key = stat_mapping[stat_type]
        try:
            return float(game_stats.get(stat_key, 0))
        except (ValueError, TypeError) as e:
            logging.error(f"Error parsing stat value: {e}")
            return None


def verify_prediction(prediction, actual_value):
    """
    Verify if a prediction was correct.
    
    Args:
        prediction: 'OVER' or 'UNDER' string
        actual_value: Actual numeric value
        line_score: Prediction line value
        
    Returns:
        Boolean indicating if prediction was correct
    """
    line_score = float(prediction['line_score'])
    
    if prediction['prediction'] == 'OVER':
        return actual_value > line_score
    else:  # 'UNDER'
        return actual_value < line_score


def update_prediction_verification(prediction_id, actual_value, is_correct):
    """
    Update the historical predictions database with verification results.
    
    Args:
        prediction_id: ID of the prediction record
        actual_value: Actual stat value
        is_correct: Boolean indicating if prediction was correct
        
    Returns:
        Boolean indicating success
    """
    try:
        with sqlite3.connect(HISTORICAL_DB) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE historical_predictions
                SET result = ?, actual_value = ?, correct = ?, verified_date = ?
                WHERE id = ?
                """,
                (
                    'HIT' if is_correct else 'MISS',
                    actual_value,
                    is_correct,
                    datetime.now().strftime('%Y-%m-%d'),
                    prediction_id
                )
            )
            conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error updating prediction verification: {e}")
        return False


def verify_predictions(days_back=7, limit=None):
    """
    Main function to verify predictions against actual game results.
    
    Args:
        days_back: Number of days back to look for unverified predictions
        limit: Optional limit on number of predictions to verify
        
    Returns:
        Dictionary with verification statistics
    """
    # Get unverified predictions
    predictions_df = get_unverified_predictions(days_back)
    
    if predictions_df.empty:
        logging.info("No unverified predictions found")
        return {"verified_count": 0, "found_games": 0, "not_found_games": 0}
    
    # Apply limit if specified
    if limit and limit > 0:
        predictions_df = predictions_df.head(limit)
    
    # Track statistics
    stats = {
        "verified_count": 0,
        "found_games": 0,
        "not_found_games": 0,
        "correct_predictions": 0,
        "processed": 0
    }
    
    # Process each prediction
    for _, prediction in predictions_df.iterrows():
        stats["processed"] += 1
        logging.info(f"Processing prediction {stats['processed']}/{len(predictions_df)}: {prediction['player_name']} {prediction['stat_type']}")
        
        # Find the game date
        game_date = get_game_date_from_prediction(
            prediction['prediction_date'], 
            prediction['team'], 
            prediction['opponent']
        )
        
        if not game_date:
            logging.warning(f"Could not find game for {prediction['player_name']} ({prediction['team']} vs {prediction['opponent']}) on or around {prediction['prediction_date']}")
            stats["not_found_games"] += 1
            continue
        
        stats["found_games"] += 1
        
        # Get player's stats for the game
        game_stats = get_player_stats_for_game(
            prediction['player_name'], 
            game_date, 
            prediction['team']
        )
        
        if not game_stats:
            logging.warning(f"Could not find game stats for {prediction['player_name']} on {game_date}")
            stats["not_found_games"] += 1
            continue
        
        # Extract the relevant stat
        actual_value = get_stat_value_from_results(game_stats, prediction['stat_type'])
        
        if actual_value is None:
            logging.warning(f"Could not extract {prediction['stat_type']} from game stats for {prediction['player_name']}")
            continue
        
        # Verify the prediction
        is_correct = verify_prediction(prediction, actual_value)
        
        # Update the database
        if update_prediction_verification(prediction['id'], actual_value, is_correct):
            stats["verified_count"] += 1
            if is_correct:
                stats["correct_predictions"] += 1
            
            logging.info(f"Verified prediction for {prediction['player_name']} {prediction['stat_type']}: " +
                         f"Predicted {prediction['prediction']} {prediction['line_score']}, " +
                         f"Actual {actual_value}, Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        # Delay to avoid hitting API rate limits
        time.sleep(1)
    
    # Calculate accuracy
    if stats["verified_count"] > 0:
        accuracy = stats["correct_predictions"] / stats["verified_count"] * 100
        stats["accuracy"] = accuracy
        logging.info(f"Overall accuracy: {accuracy:.2f}% ({stats['correct_predictions']}/{stats['verified_count']})")
    
    return stats


def main():
    """Main function to run the verification process."""
    parser = argparse.ArgumentParser(description='Verify NBA prediction accuracy against actual game results')
    parser.add_argument('--days', type=int, default=7, help='Number of days back to check predictions')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of predictions to verify')
    args = parser.parse_args()
    
    logging.info(f"Starting prediction verification for the past {args.days} days")
    
    try:
        stats = verify_predictions(days_back=args.days, limit=args.limit)
        
        # Print summary
        print("\nVerification Summary:")
        print(f"Processed: {stats['processed']} predictions")
        print(f"Found games: {stats['found_games']}")
        print(f"Not found games: {stats['not_found_games']}")
        print(f"Verified: {stats['verified_count']} predictions")
        
        if 'accuracy' in stats:
            print(f"Accuracy: {stats['accuracy']:.2f}% ({stats['correct_predictions']}/{stats['verified_count']})")
        
        logging.info("Prediction verification completed")
        
    except Exception as e:
        logging.error(f"Error during verification process: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())