import requests
import sqlite3
import concurrent.futures
import logging
import time
import argparse
import os
import sys
from datetime import datetime
from tqdm.auto import tqdm  # Using tqdm.auto for better terminal compatibility
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Set up logging
os.makedirs("/Users/lukesmac/Models/nba/logs", exist_ok=True)
logging.basicConfig(
    filename=f"/Users/lukesmac/Models/nba/logs/game_stats_fetcher.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API details
url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
headers = {
    "x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
    "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
}

# Database setup
DB_PATH = '/Users/lukesmac/Models/nba/data/player_game_stats.db'

# Fantasy points scoring system
querystring_base = {
    "fantasyPoints": "true",
    "pts": "1",
    "reb": "1.25", 
    "stl": "3",
    "blk": "3",
    "ast": "1.5",
    "TOV": "-1",
    "mins": "0",
    "doubleDouble": "0",
    "tripleDouble": "0",
    "quadDouble": "0"
}

@dataclass
class GameStats:
    """Data class for game statistics."""
    player_id: str
    game_id: str
    game_date: str
    team: str
    opponent: str
    home: bool
    win: bool
    minutes: float
    points: int
    assists: int
    rebounds: int
    steals: int
    blocks: int
    turnovers: int
    three_pointers_made: int
    field_goals_made: int
    field_goals_attempted: int
    free_throws_made: int
    free_throws_attempted: int
    fantasy_points: float

def setup_database() -> None:
    """Set up the SQLite database and create necessary tables.
    
    Creates the player_game_stats table if it doesn't exist, with appropriate
    columns for storing game statistics. Drops existing table to ensure clean state.
    """
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Drop existing table if it exists
        cursor.execute('DROP TABLE IF EXISTS player_game_stats')
        
        # Create the player_game_stats table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            game_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            team TEXT,
            opponent TEXT,
            home BOOLEAN,
            win BOOLEAN,
            minutes REAL,
            points INTEGER,
            assists INTEGER,
            rebounds INTEGER,
            steals INTEGER,
            blocks INTEGER,
            turnovers INTEGER,
            three_pointers_made INTEGER,
            field_goals_made INTEGER,
            field_goals_attempted INTEGER,
            free_throws_made INTEGER, 
            free_throws_attempted INTEGER,
            fantasy_points REAL,
            UNIQUE(player_id, game_id)
        )
        ''')
        
        conn.commit()
        logging.info("Database setup complete")
        
    except sqlite3.Error as e:
        logging.error(f"Database setup error: {e}")
        raise
        
    finally:
        conn.close()

def get_player_ids() -> List[Tuple[str, str]]:
    """Get all player IDs and names from the database.
    
    Returns:
        List[Tuple[str, str]]: List of tuples containing (player_id, player_name)
    """
    connection = sqlite3.connect('/Users/lukesmac/Models/nba_players.db')
    cursor = connection.cursor()
    
    try:
        cursor.execute("SELECT playerID, name FROM players")
        players = cursor.fetchall()
        logging.info(f"Retrieved {len(players)} players from database")
        return players
    
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []
    
    finally:
        connection.close()

def get_existing_game_ids(player_id: str) -> set:
    """Get existing game IDs for a player to avoid duplicates."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT game_id FROM player_game_stats WHERE player_id = ?",
            (player_id,)
        )
        existing_game_ids = {row[0] for row in cursor.fetchall()}
        return existing_game_ids
    
    except sqlite3.Error as e:
        logging.error(f"Error fetching existing game IDs: {e}")
        return set()
    
    finally:
        conn.close()

def fetch_player_games(player_data: Tuple[str, str], season: str) -> Dict[str, Any]:
    """Fetch game stats for a specific player and season.
    
    Checks if player already has stats for the season before making API call.
    """
    player_id, player_name = player_data
    
    # Ensure player_id is a string
    player_id = str(player_id)
    
    # Check if we already have game data for this player
    existing_game_ids = get_existing_game_ids(player_id)
    
    # Skip API call if player already has a substantial number of games for this season
    # NBA regular season has 82 games, so if we have more than 75, likely complete
    if len(existing_game_ids) > 75:
        logging.debug(f"Skipping {player_name} (ID: {player_id}) - already have {len(existing_game_ids)} games")
        return {
            "player_id": player_id, 
            "player_name": player_name,
            "success": True, 
            "skipped": True, 
            "existing_games": len(existing_game_ids)
        }
    
    # Create querystring exactly like the example
    querystring = {
        "playerID": player_id,
        "season": str(2022),
        "fantasyPoints": "true",
        "pts": "1",
        "reb": "1.25",
        "stl": "3",
        "blk": "3",
        "ast": "1.5",
        "TOV": "-1",
        "mins": "0",
        "doubleDouble": "0",
        "tripleDouble": "0",
        "quadDouble": "0"
    }
    
    try:
        # Add slight delay to avoid rate limits
        time.sleep(0.1)
        logging.debug(f"Fetching data for {player_name} (ID: {player_id})")
        logging.debug(f"Query parameters: {querystring}")
        
        response = requests.get(url, headers=headers, params=querystring)
        response_text = response.text
        
        if response.status_code == 200:
            try:
                data = response.json()
                logging.debug(f"API Response for {player_name}: {data}")
                if 'body' not in data:
                    logging.error(f"Invalid API response format for {player_name} (ID: {player_id}): {data}")
                    return {"player_id": player_id, "success": False, "error": "Invalid API response format"}
                
                # Add existing_game_ids to the result so we can filter them later
                return {
                    "player_id": player_id, 
                    "player_name": player_name, 
                    "data": data, 
                    "success": True, 
                    "existing_game_ids": existing_game_ids
                }
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON for {player_name} (ID: {player_id}): {e}\nResponse: {response_text}")
                return {"player_id": player_id, "success": False, "error": f"JSON parse error: {str(e)}"}
        else:
            logging.error(f"API request failed for {player_name} (ID: {player_id}): Status {response.status_code}\nResponse: {response_text}")
            return {"player_id": player_id, "success": False, "error": f"Status code: {response.status_code}"}
            
    except Exception as e:
        logging.error(f"Exception while fetching data for {player_name} (ID: {player_id}): {str(e)}")
        return {"player_id": player_id, "success": False, "error": str(e)}

def parse_game_data(game_id: str, game_data: Dict[str, Any], player_id: str) -> Optional[Dict[str, Any]]:
    """Parse game data from API response into database format.
    
    Args:
        game_id: The unique identifier for the game (format: YYYYMMDD_AWAY@HOME)
        game_data: Dictionary containing game statistics from the API
        player_id: The player's unique identifier
    
    Returns:
        Optional[Dict[str, Any]]: Parsed game data dictionary or None if parsing fails
    """
    try:
        # Extract game date and teams from game ID
        date_str, teams = game_id.split('_')
        away_team, home_team = teams.split('@')
        game_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        
        # Determine if player's team was home or away
        team = game_data.get('team', '')
        is_home = team == home_team
        opponent = home_team if team == away_team else away_team
        
        # Convert stats to appropriate types with safe conversion
        def safe_convert(value: Any, convert_func: callable, default: Any = 0) -> Any:
            """Safely convert values with proper error handling."""
            try:
                return convert_func(value or default)
            except (ValueError, TypeError):
                return default
        
        return GameStats(
            player_id=player_id,
            game_id=game_id,
            game_date=game_date,
            team=team,
            opponent=opponent,
            home=is_home,
            win=False,  # We'll need additional data to determine this
            minutes=safe_convert(game_data.get('mins'), float),
            points=safe_convert(game_data.get('pts'), int),
            assists=safe_convert(game_data.get('ast'), int),
            rebounds=safe_convert(game_data.get('reb'), int),
            steals=safe_convert(game_data.get('stl'), int),
            blocks=safe_convert(game_data.get('blk'), int),
            turnovers=safe_convert(game_data.get('TOV'), int),
            three_pointers_made=safe_convert(game_data.get('tptfgm'), int),
            field_goals_made=safe_convert(game_data.get('fgm'), int),
            field_goals_attempted=safe_convert(game_data.get('fga'), int),
            free_throws_made=safe_convert(game_data.get('ftm'), int),
            free_throws_attempted=safe_convert(game_data.get('fta'), int),
            fantasy_points=safe_convert(game_data.get('fantasyPoints'), float)
        ).__dict__
        
    except Exception as e:
        logging.error(f"Error parsing game data: {str(e)}")
        return None

def store_player_game_stats(stats_data: Dict[str, Any]) -> int:
    """Store player game stats in the SQLite database.
    
    Args:
        stats_data: Dictionary containing player stats from API response
    
    Returns:
        int: Number of games added to the database
    """
    # If player was skipped because they already had stats, return 0 new games
    if stats_data.get("skipped", False):
        return 0
        
    if not stats_data.get("success", False):
        return 0
    
    player_id = stats_data["player_id"]
    data = stats_data.get("data", {})
    existing_game_ids = stats_data.get("existing_game_ids", set())
    
    if 'body' not in data:
        return 0
    
    # Get game data - each key in body is a game ID
    games_data = data['body']
    if not isinstance(games_data, dict) or not games_data:
        return 0
    
    # Parse games into database format, filtering out existing ones
    parsed_games = []
    new_games_count = 0
    skipped_games_count = 0
    
    for game_id, game_data in games_data.items():
        # Skip this game if it's already in the database
        if game_id in existing_game_ids:
            skipped_games_count += 1
            continue
            
        parsed_game = parse_game_data(game_id, game_data, player_id)
        if parsed_game:
            parsed_games.append(parsed_game)
            new_games_count += 1
    
    if not parsed_games:
        # No new games to add
        logging.debug(f"No new games to add for player {player_id}. Skipped {skipped_games_count} existing games.")
        return 0
    
    # Insert only new games into database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        for game in parsed_games:
            cursor.execute('''
            INSERT INTO player_game_stats 
            (player_id, game_id, game_date, team, opponent, home, win, 
            minutes, points, assists, rebounds, steals, blocks, turnovers, 
            three_pointers_made, field_goals_made, field_goals_attempted, 
            free_throws_made, free_throws_attempted, fantasy_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game['player_id'], game['game_id'], game['game_date'], game['team'], 
                game['opponent'], game['home'], game['win'],
                game['minutes'], game['points'], game['assists'], game['rebounds'], 
                game['steals'], game['blocks'], game['turnovers'], 
                game['three_pointers_made'], game['field_goals_made'], game['field_goals_attempted'],
                game['free_throws_made'], game['free_throws_attempted'],
                game['fantasy_points']
            ))
        
        conn.commit()
        logging.debug(f"Added {new_games_count} new games for player {player_id}. Skipped {skipped_games_count} existing games.")
        return new_games_count
    
    except sqlite3.Error as e:
        logging.error(f"Database error while storing game stats: {e}")
        conn.rollback()
        return 0
    
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description='Fetch NBA player game stats.')
    parser.add_argument('--seasons', nargs='+', default=['2024'], 
                        help='List of seasons to fetch (e.g., 2023 2024)')
    parser.add_argument('--max-workers', type=int, default=10, 
                        help='Maximum number of parallel workers')
    parser.add_argument('--player-limit', type=int, default=None, 
                        help='Limit number of players to process (for testing)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--recreate-db', action='store_true',
                        help='Drop and recreate the database tables (warning: all data will be lost)')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    print(f"Starting player game stats fetching process")
    
    # Setup database - only recreate if explicitly requested
    if args.recreate_db:
        setup_database()
        print("Database tables recreated (all existing data was deleted)")
    else:
        # Ensure database exists without dropping tables
        db_path = Path(DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Create the player_game_stats table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_game_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                team TEXT,
                opponent TEXT,
                home BOOLEAN,
                win BOOLEAN,
                minutes REAL,
                points INTEGER,
                assists INTEGER,
                rebounds INTEGER,
                steals INTEGER,
                blocks INTEGER,
                turnovers INTEGER,
                three_pointers_made INTEGER,
                field_goals_made INTEGER,
                field_goals_attempted INTEGER,
                free_throws_made INTEGER, 
                free_throws_attempted INTEGER,
                fantasy_points REAL,
                UNIQUE(player_id, game_id)
            )
            ''')
            
            conn.commit()
            logging.info("Database checked - using existing tables")
            
        except sqlite3.Error as e:
            logging.error(f"Database setup error: {e}")
            raise
            
        finally:
            conn.close()
    
    # Get player IDs from database
    players = get_player_ids()
    if args.player_limit and args.player_limit < len(players):
        players = players[:args.player_limit]
        print(f"Limiting to {args.player_limit} players")
    
    if not players:
        print("No players found. Exiting.")
        return
    
    total_games_added = 0
    total_players_processed = 0
    total_players_skipped = 0
    total_players_with_new_games = 0
    
    # Create a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for season in args.seasons:
            print(f"\nProcessing season {season}")
            season_games_added = 0
            season_players_skipped = 0
            season_players_with_new_games = 0
            
            # Submit tasks for each player
            futures = []
            for player in players:
                futures.append(executor.submit(fetch_player_games, player, season))
            
            # Create a progress bar that properly stays on one line
            total_players = len(futures)
            progress_format = "Season {season} [{updated} players updated, {games} new games] |{bar}| {percentage:3.0f}% ({n}/{total}) [{elapsed}<{remaining}]"
            progress_desc = progress_format.format(
                season=season,
                updated=0,
                games=0,
                bar=" "*20,
                percentage=0,
                n=0,
                total=total_players,
                elapsed="0:00",
                remaining="??:??"
            )
            
            # Initialize progress bar with minimal refresh rate
            with tqdm(
                total=total_players,
                desc=progress_desc,
                unit="",
                dynamic_ncols=True,
                position=0,
                leave=True,
                file=sys.stdout,
                ascii=True, # Using ASCII for better compatibility
                mininterval=1.0,  # Only refresh display once per second max
            ) as progress_bar:
                # Process results as they complete
                completed = 0
                
                # Results processing
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        # Only update the progress bar, not the description yet
                        progress_bar.update(1)
                        
                        # Update stats display only occasionally to avoid too many refreshes
                        if completed % 10 == 0 or completed == total_players:
                            # Process all results collected so far but not processed
                            for pending_result in results:
                                player_name = pending_result.get("player_name", "Unknown")
                                
                                # Handle case where player was skipped
                                if pending_result.get("skipped", False):
                                    season_players_skipped += 1
                                    total_players_skipped += 1
                                    logging.debug(f"Skipped {player_name} - already has sufficient data")
                                elif pending_result.get("success", False):
                                    games_added = store_player_game_stats(pending_result)
                                    if games_added > 0:
                                        season_players_with_new_games += 1
                                        total_players_with_new_games += 1
                                        season_games_added += games_added
                                        total_games_added += games_added
                                
                                total_players_processed += 1
                            
                            # Clear the processed results
                            results = []
                            
                            # Update the progress bar description with new stats
                            new_desc = progress_format.format(
                                season=season,
                                updated=season_players_with_new_games,
                                games=season_games_added,
                                bar=" "*20,  # This will be replaced by tqdm
                                percentage=int(100 * completed / total_players),
                                n=completed,
                                total=total_players,
                                elapsed=progress_bar.format_dict.get('elapsed', '0:00'),
                                remaining=progress_bar.format_dict.get('remaining', '??:??')
                            )
                            progress_bar.set_description(new_desc)
                            
                    except Exception as e:
                        logging.error(f"Exception while processing results: {str(e)}")
                        completed += 1
                        progress_bar.update(1)
                        
                # Process any remaining results
                for pending_result in results:
                    player_name = pending_result.get("player_name", "Unknown")
                    
                    # Handle case where player was skipped
                    if pending_result.get("skipped", False):
                        season_players_skipped += 1
                        total_players_skipped += 1
                        logging.debug(f"Skipped {player_name} - already has sufficient data")
                    elif pending_result.get("success", False):
                        games_added = store_player_game_stats(pending_result)
                        if games_added > 0:
                            season_players_with_new_games += 1
                            total_players_with_new_games += 1
                            season_games_added += games_added
                            total_games_added += games_added
                    
                    total_players_processed += 1
            
            print(f"Season {season} complete:")
            print(f"- Players with new games: {season_players_with_new_games}")
            print(f"- Players skipped (already had data): {season_players_skipped}")
            print(f"- Total new games added: {season_games_added}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nProcess Summary:")
    print(f"- Added {total_games_added} new games")
    print(f"- Updated data for {total_players_with_new_games} players")
    print(f"- Skipped {total_players_skipped} players (already had sufficient data)")
    print(f"- Process completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()