import requests
import sqlite3
import json
from typing import Dict, Tuple, List
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

def convert_ml_odds(odds_str: str) -> int:
    """Convert moneyline odds string to integer format, handling 'even' case."""
    if odds_str == "even":
        return 100
    return int(odds_str)

def get_unique_game_ids() -> List[str]:
    """Get all unique game IDs from the database."""
    conn = sqlite3.connect('/Users/lukesmac/Models/nba/data/player_game_stats.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT game_id 
        FROM player_game_stats 
        WHERE avg_home_ML_odds IS NULL 
        OR avg_away_ML_odds IS NULL
        ORDER BY game_date DESC
    """)
    results = cursor.fetchall()
    
    conn.close()
    return [result[0] for result in results]

def fetch_odds(game_id: str) -> Tuple[str, Dict]:
    """Fetch odds from the API for a specific game ID."""
    try:
        url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBABettingOdds"
        headers = {
            "x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
            "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
        }
        params = {"gameID": game_id, "itemFormat": "map"}
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return game_id, None
            
        data = response.json()
        if 'body' not in data or not data['body']:
            return game_id, None
            
        return game_id, data
    except Exception as e:
        print(f"Error fetching odds for {game_id}: {e}")
        return game_id, None

def calculate_ml_odds(odds_data: Dict) -> Tuple[float, float]:
    """Calculate average moneyline odds for home and away teams."""
    if not odds_data or 'body' not in odds_data:
        return None, None
        
    try:
        game_key = list(odds_data['body'].keys())[0]
        game_odds = odds_data['body'][game_key]
        
        bookmakers = ['betmgm', 'bet365', 'fanduel', 'hardrock', 'espnbet', 'betrivers', 'draftkings']
        away_odds = []
        home_odds = []
        
        for bookmaker in bookmakers:
            if bookmaker in game_odds:
                away_ml = game_odds[bookmaker].get('awayTeamMLOdds')
                home_ml = game_odds[bookmaker].get('homeTeamMLOdds')
                
                if away_ml and home_ml:
                    try:
                        away_odds.append(convert_ml_odds(away_ml))
                        home_odds.append(convert_ml_odds(home_ml))
                    except ValueError:
                        continue
        
        if not away_odds or not home_odds:
            return None, None
            
        return statistics.mean(away_odds), statistics.mean(home_odds)
    except Exception:
        return None, None

def batch_update_database(updates: List[Tuple[float, float, str]]) -> None:
    """Update the database with multiple game odds at once."""
    if not updates:
        return
        
    conn = sqlite3.connect('/Users/lukesmac/Models/nba/data/player_game_stats.db')
    cursor = conn.cursor()
    
    try:
        cursor.executemany("""
            UPDATE player_game_stats 
            SET avg_home_ML_odds = ?, avg_away_ML_odds = ?
            WHERE game_id = ?
        """, updates)
        
        conn.commit()
        print(f"Successfully updated {len(updates)} games in database")
    except Exception as e:
        print(f"Error during batch database update: {e}")
    finally:
        conn.close()

def process_game(game_id: str) -> Tuple[str, float, float]:
    """Process a single game and return the odds."""
    game_id, odds_data = fetch_odds(game_id)
    if odds_data is None:
        return game_id, None, None
        
    avg_away_odds, avg_home_odds = calculate_ml_odds(odds_data)
    return game_id, avg_away_odds, avg_home_odds

def process_all_games():
    """Process all games that need odds updates using parallel processing."""
    game_ids = get_unique_game_ids()
    total_games = len(game_ids)
    print(f"Found {total_games} games to process")
    
    # Process games in parallel
    updates = []
    processed = 0
    batch_size = 10  # Process in batches of 10 games
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        for game_id, away_odds, home_odds in executor.map(process_game, game_ids):
            processed += 1
            
            if away_odds is not None and home_odds is not None:
                updates.append((home_odds, away_odds, game_id))
                
            # Update progress
            if processed % 10 == 0 or processed == total_games:
                print(f"Processed {processed}/{total_games} games")
            
            # Batch update database
            if len(updates) >= batch_size:
                batch_update_database(updates)
                updates = []
    
    # Final batch update
    if updates:
        batch_update_database(updates)

def main():
    """Main function to run the odds processing."""
    print("Starting odds processing...")
    start_time = datetime.now()
    
    process_all_games()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nFinished processing all games in {duration:.1f} seconds")

if __name__ == "__main__":
    main()