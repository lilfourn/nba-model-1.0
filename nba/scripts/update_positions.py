#!/usr/bin/env python3
"""
update_positions.py - Fetch NBA depth charts and update player positions in the database

This script retrieves current NBA depth charts and adds position information 
to player_game_stats in the database, enhancing the dataset for ML modeling.

Usage:
    python update_positions.py [--db-path DB_PATH]
"""

import requests
import sqlite3
import logging
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Configure logging
os.makedirs('/Users/lukesmac/Models/nba/logs', exist_ok=True)
logging.basicConfig(
    filename=f'/Users/lukesmac/Models/nba/logs/update_positions_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database path
DEFAULT_DB_PATH = '/Users/lukesmac/Models/nba/data/player_game_stats.db'

# API details
DEPTH_CHART_URL = "https://tank01-fantasy-stats.p.rapidapi.com/getNBADepthCharts"
HEADERS = {
    "x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
    "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
}

# NBA positions
NBA_POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']

def connect_to_db(db_path):
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        logging.info(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def fetch_depth_charts():
    """Fetch NBA depth charts from the API."""
    try:
        logging.info("Fetching NBA depth charts from API")
        response = requests.get(DEPTH_CHART_URL, headers=HEADERS)
        
        # Add a log for response text before JSON parsing
        logging.debug(f"API Response text: {response.text[:500]}...")  # Log first 500 chars
        
        if response.status_code == 200:
            # Try parsing as JSON
            try:
                data = response.json()
                logging.info("Successfully retrieved depth charts")
                return data
            except json.JSONDecodeError as e:
                # If JSON parsing fails, log the issue and create a structured response manually
                logging.error(f"Failed to parse JSON response: {e}")
                logging.info("Attempting manual parsing of the response")
                
                # For debugging, save the raw response to a file
                with open('/Users/lukesmac/Models/nba/data/depth_charts_raw.txt', 'w') as f:
                    f.write(response.text)
                
                # Return the raw response as is for further processing
                return {"statusCode": response.status_code, "body": response.text}
        else:
            logging.error(f"API request failed with status code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            raise Exception(f"API request failed: {response.status_code}")
    
    except Exception as e:
        logging.error(f"Error fetching depth charts: {e}")
        raise

def parse_depth_charts(depth_data):
    """
    Parse the depth chart data and extract player positions.
    
    Returns:
        dict: Mapping of player IDs to their primary position and depth
    """
    player_positions = {}
    
    try:
        # For debugging
        logging.info(f"Response format: {json.dumps(depth_data, indent=2)[:200]}...")
        
        # Extract the body from the response
        if 'body' not in depth_data:
            logging.error("No 'body' key in API response")
            return player_positions
        
        body = depth_data['body']
        
        # Handle different JSON structures
        if isinstance(body, dict):
            # Handle case where body is a dict with team codes as keys
            for team_code, team_data in body.items():
                if not isinstance(team_data, dict) or 'depthChart' not in team_data:
                    logging.debug(f"No depthChart for team {team_code}")
                    continue
                
                depth_chart = team_data['depthChart']
                team_name = team_data.get('teamName', team_code)
                
                # Process each position in depth chart
                for position, players in depth_chart.items():
                    if position not in NBA_POSITIONS:
                        continue
                    
                    if not isinstance(players, list):
                        logging.debug(f"Position {position} for team {team_code} is not a list")
                        continue
                    
                    for idx, player in enumerate(players):
                        if not isinstance(player, dict) or 'playerID' not in player:
                            continue
                        
                        player_id = player['playerID']
                        player_name = player.get('longName', 'Unknown')
                        depth_position = player.get('depthPosition', f"{position}{idx+1}")
                        
                        # Extract depth number
                        try:
                            depth = int(''.join(filter(str.isdigit, depth_position)))
                        except:
                            depth = idx + 1
                        
                        # Store player position data
                        if player_id not in player_positions or depth < player_positions[player_id]['depth']:
                            player_positions[player_id] = {
                                'position': position,
                                'depth': depth,
                                'name': player_name,
                                'team': team_name
                            }
        elif isinstance(body, list):
            # Handle case where body is a list of teams
            for team_idx, team_data in enumerate(body):
                if not isinstance(team_data, dict) or 'depthChart' not in team_data:
                    continue
                
                depth_chart = team_data['depthChart']
                team_name = team_data.get('teamName', f"Team{team_idx}")
                
                # Process positions
                for position in NBA_POSITIONS:
                    if position not in depth_chart:
                        continue
                    
                    position_players = depth_chart[position]
                    if not isinstance(position_players, list):
                        continue
                    
                    for idx, player in enumerate(position_players):
                        if not isinstance(player, dict) or 'playerID' not in player:
                            continue
                        
                        player_id = player['playerID']
                        player_name = player.get('longName', 'Unknown')
                        depth = idx + 1
                        
                        if player_id not in player_positions or depth < player_positions[player_id]['depth']:
                            player_positions[player_id] = {
                                'position': position,
                                'depth': depth,
                                'name': player_name,
                                'team': team_name
                            }
        else:
            # Direct access to the first team
            # Based on the example output structure
            logging.info("Trying direct access to body structure")
            for team_idx in range(30):  # Loop through possible team indices (usually 0-29 for NBA teams)
                team_key = str(team_idx)
                if team_key not in body:
                    continue
                
                team_data = body[team_key]
                if 'depthChart' not in team_data:
                    continue
                
                depth_chart = team_data['depthChart']
                team_name = team_data.get('teamName', f"Team{team_idx}")
                
                for position in NBA_POSITIONS:
                    if position not in depth_chart:
                        continue
                    
                    position_players = depth_chart[position]
                    if not isinstance(position_players, dict):
                        continue
                    
                    # Loop through player indices in the position
                    for player_idx in range(15):  # Max 15 players per position
                        idx_key = str(player_idx)
                        if idx_key not in position_players:
                            continue
                        
                        player = position_players[idx_key]
                        if not isinstance(player, dict) or 'playerID' not in player:
                            continue
                        
                        player_id = player['playerID']
                        player_name = player.get('longName', 'Unknown')
                        depth_position = player.get('depthPosition', f"{position}{player_idx+1}")
                        
                        # Extract depth number
                        try:
                            depth = int(''.join(filter(str.isdigit, depth_position)))
                        except:
                            depth = player_idx + 1
                        
                        # Store player position
                        if player_id not in player_positions or depth < player_positions[player_id]['depth']:
                            player_positions[player_id] = {
                                'position': position,
                                'depth': depth,
                                'name': player_name,
                                'team': team_name
                            }
                            logging.debug(f"Added player: {player_name} ({player_id}) - {position}{depth}")
        
        if not player_positions:
            # If we still couldn't parse, let's try a more flexible approach
            logging.info("Attempting flexible parsing approach for depth chart data")
            # Print first 1000 chars of the depth_data for debugging
            logging.debug(f"Depth data sample: {str(depth_data)[:1000]}")
            
            # Try to iterate through any structure
            def recursive_find_players(data, team="Unknown", pos=None, depth=1):
                if isinstance(data, dict):
                    if 'playerID' in data and 'longName' in data:
                        player_id = data['playerID']
                        player_name = data['longName']
                        position = pos
                        
                        if not position:
                            # Try to extract position from depthPosition
                            if 'depthPosition' in data:
                                depth_pos = data['depthPosition']
                                for p in NBA_POSITIONS:
                                    if p in depth_pos:
                                        position = p
                                        try:
                                            depth = int(''.join(filter(str.isdigit, depth_pos)))
                                        except:
                                            pass
                                        break
                            
                        if position and (player_id not in player_positions or depth < player_positions[player_id]['depth']):
                            player_positions[player_id] = {
                                'position': position,
                                'depth': depth,
                                'name': player_name,
                                'team': team
                            }
                            logging.debug(f"Found player: {player_name} ({player_id}) - {position}{depth}")
                    
                    # Current key might be a position
                    current_pos = pos
                    for key in data.keys():
                        if key in NBA_POSITIONS:
                            current_pos = key
                            
                        # Recursively process nested data
                        if isinstance(data[key], (dict, list)):
                            recursive_find_players(data[key], team, current_pos, depth)
                
                elif isinstance(data, list):
                    for idx, item in enumerate(data):
                        recursive_find_players(item, team, pos, idx + 1)
            
            # Start recursive search through the data
            recursive_find_players(depth_data)
        
        logging.info(f"Parsed positions for {len(player_positions)} players")
        for pid, pdata in list(player_positions.items())[:5]:  # Log first 5 for debugging
            logging.info(f"Player {pdata['name']} ({pid}): {pdata['position']}{pdata['depth']}")
        
        return player_positions
    
    except Exception as e:
        logging.error(f"Error parsing depth charts: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return player_positions

def get_existing_players(conn):
    """
    Get a list of player IDs that exist in the database.
    
    Returns:
        set: Set of player IDs in the database
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT player_id FROM player_game_stats")
        player_ids = {row[0] for row in cursor.fetchall()}
        logging.info(f"Found {len(player_ids)} unique players in database")
        return player_ids
    except sqlite3.Error as e:
        logging.error(f"Error getting existing players: {e}")
        return set()

def update_player_positions(conn, player_positions):
    """
    Update player positions in the database.
    
    Args:
        conn: SQLite connection
        player_positions: Dictionary of player IDs and their positions
    
    Returns:
        int: Number of players updated
    """
    try:
        # Check if position column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(player_game_stats)")
        columns = {col[1] for col in cursor.fetchall()}
        
        # Add position column if it doesn't exist
        if 'position' not in columns:
            logging.info("Adding 'position' column to player_game_stats table")
            cursor.execute("ALTER TABLE player_game_stats ADD COLUMN position TEXT")
        
        # Add position_depth column if it doesn't exist
        if 'position_depth' not in columns:
            logging.info("Adding 'position_depth' column to player_game_stats table")
            cursor.execute("ALTER TABLE player_game_stats ADD COLUMN position_depth TEXT")
        
        # Get existing player IDs
        existing_players = get_existing_players(conn)
        
        # Match players and update positions
        updated_count = 0
        updated_rows = 0
        skipped_players = 0
        
        # First, verify how many rows will be updated
        for player_id, player_data in player_positions.items():
            if player_id in existing_players:
                position = player_data['position']
                depth = player_data['depth']
                
                # Create position_depth string (e.g., "SF1", "C2")
                position_depth = f"{position}{depth}"
                
                # Check how many rows will be updated for this player
                cursor.execute(
                    "SELECT COUNT(*) FROM player_game_stats WHERE player_id = ?",
                    (player_id,)
                )
                row_count = cursor.fetchone()[0]
                
                # Check if player already has positions set
                cursor.execute(
                    "SELECT COUNT(*) FROM player_game_stats WHERE player_id = ? AND position IS NOT NULL AND position_depth IS NOT NULL",
                    (player_id,)
                )
                existing_positions = cursor.fetchone()[0]
                
                # If this player already has positions set for all their games, skip
                if existing_positions == row_count:
                    skipped_players += 1
                    continue
                
                # Update all records for this player
                logging.debug(f"Updating {row_count} rows for player {player_id} ({player_data['name']}) with position {position_depth}")
                cursor.execute(
                    """
                    UPDATE player_game_stats 
                    SET position = ?, position_depth = ? 
                    WHERE player_id = ? AND (position IS NULL OR position = '' OR position_depth IS NULL OR position_depth = '')
                    """,
                    (position, position_depth, player_id)
                )
                updated_rows += cursor.rowcount
                updated_count += 1
        
        # Commit changes
        conn.commit()
        logging.info(f"Updated positions for {updated_count} players, modifying {updated_rows} rows")
        logging.info(f"Skipped {skipped_players} players that already had positions set")
        
        # Verify the update was successful
        cursor.execute("SELECT COUNT(*) FROM player_game_stats WHERE position IS NOT NULL AND position_depth IS NOT NULL")
        total_rows_with_position = cursor.fetchone()[0]
        logging.info(f"Total rows with position after update: {total_rows_with_position}")
        
        # Create a player positions table for reference
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_positions (
            player_id TEXT PRIMARY KEY,
            name TEXT,
            position TEXT,
            position_depth TEXT,
            depth INTEGER,
            team TEXT,
            updated_at TEXT
        )
        """)
        
        # Insert or update player position data
        current_time = datetime.now().isoformat()
        for player_id, player_data in player_positions.items():
            position = player_data['position']
            depth = player_data['depth']
            position_depth = f"{position}{depth}"
            
            cursor.execute("""
            INSERT OR REPLACE INTO player_positions
            (player_id, name, position, position_depth, depth, team, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                player_id,
                player_data['name'],
                position,
                position_depth,
                depth,
                player_data['team'],
                current_time
            ))
        
        conn.commit()
        logging.info(f"Updated player_positions reference table with {len(player_positions)} entries")
        
        # Print a few sample rows to verify the update
        cursor.execute("""
        SELECT player_id, position, position_depth, game_date FROM player_game_stats 
        WHERE position IS NOT NULL AND position_depth IS NOT NULL
        LIMIT 5
        """)
        sample_rows = cursor.fetchall()
        for row in sample_rows:
            logging.debug(f"Sample row after update: {row}")
        
        return updated_count
    
    except sqlite3.Error as e:
        logging.error(f"Database error updating positions: {e}")
        conn.rollback()
        return 0

def save_raw_depth_charts(depth_data):
    """Save the raw depth chart data for reference"""
    try:
        data_dir = Path('/Users/lukesmac/Models/nba/data')
        data_dir.mkdir(exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = data_dir / f"depth_charts_{timestamp}.json"
        
        with open(file_path, 'w') as f:
            json.dump(depth_data, f, indent=2)
            
        logging.info(f"Saved raw depth chart data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save raw depth chart data: {e}")

def create_manual_test_data():
    """
    Create test data based on the example output in the prompt.
    This is useful for testing the parsing logic when the API fails.
    """
    # Example data structure from the prompt
    test_data = {
        "statusCode": 200,
        "body": {
            "0": {
                "depthChart": {
                    "SF": {
                        "0": {
                            "depthPosition": "SF1",
                            "longName": "Zaccharie Risacher",
                            "playerID": "943541715989"
                        },
                        "1": {
                            "depthPosition": "SF2",
                            "longName": "Vit Krejci",
                            "playerID": "941044811139"
                        },
                        "2": {
                            "depthPosition": "SF3",
                            "longName": "Caris LeVert",
                            "playerID": "28838048249"
                        },
                        "3": {
                            "depthPosition": "SF4",
                            "longName": "Terance Mann",
                            "playerID": "28558319499"
                        },
                        "4": {
                            "depthPosition": "SF5",
                            "longName": "Georges Niang",
                            "playerID": "28428786249"
                        }
                    },
                    "C": {
                        "0": {
                            "depthPosition": "C1",
                            "longName": "Onyeka Okongwu",
                            "playerID": "94054452027"
                        },
                        "1": {
                            "depthPosition": "C2",
                            "longName": "Clint Capela",
                            "playerID": "28218011729"
                        },
                        "2": {
                            "depthPosition": "C3",
                            "longName": "Dom Barlow",
                            "playerID": "948647045669"
                        },
                        "3": {
                            "depthPosition": "C4",
                            "longName": "Mouhamed Gueye",
                            "playerID": "943649135539"
                        }
                    },
                    # Add other positions as needed
                    "PG": {
                        "0": {
                            "depthPosition": "PG1",
                            "longName": "Trae Young",
                            "playerID": "28558188249"
                        }
                    },
                    "SG": {
                        "0": {
                            "depthPosition": "SG1",
                            "longName": "Bogdan Bogdanovic",
                            "playerID": "28328781569"
                        }
                    },
                    "PF": {
                        "0": {
                            "depthPosition": "PF1",
                            "longName": "Jalen Johnson",
                            "playerID": "940544521589"
                        }
                    }
                },
                "teamName": "Atlanta Hawks"
            }
            # Add more teams if needed for testing
        }
    }
    
    return test_data

def main():
    """Main function to update player positions"""
    parser = argparse.ArgumentParser(description='Fetch NBA depth charts and update player positions')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH,
                      help=f'Path to player game stats database (default: {DEFAULT_DB_PATH})')
    parser.add_argument('--test-mode', action='store_true',
                      help='Use test data instead of querying the API')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--dump-response', action='store_true',
                      help='Dump the API response to a file and exit')
    parser.add_argument('--save-json', action='store_true',
                      help='Save raw depth chart data as JSON files')
    parser.add_argument('--force', action='store_true',
                      help='Force update all positions even if they already exist')
    parser.add_argument('--verify', action='store_true',
                      help='Verify database positions without updating')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Add a console handler for debug mode
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
    
    try:
        # Verify option - just check the database
        if args.verify:
            conn = connect_to_db(args.db_path)
            cursor = conn.cursor()
            
            # Check for position column
            cursor.execute("PRAGMA table_info(player_game_stats)")
            columns = {col[1] for col in cursor.fetchall()}
            if 'position' not in columns:
                print("Position column does not exist in player_game_stats table.")
                return
                
            # Get stats on position values
            cursor.execute("SELECT COUNT(*) FROM player_game_stats")
            total_rows = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM player_game_stats WHERE position IS NOT NULL")
            filled_rows = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_game_stats")
            total_players = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT player_id) FROM player_game_stats WHERE position IS NOT NULL")
            players_with_position = cursor.fetchone()[0]
            
            # Sample positions
            cursor.execute("""
            SELECT player_id, position, position_depth, COUNT(*) as count 
            FROM player_game_stats 
            WHERE position IS NOT NULL 
            GROUP BY player_id, position, position_depth 
            ORDER BY count DESC 
            LIMIT 10
            """)
            position_samples = cursor.fetchall()
            
            print(f"Database Position Statistics:")
            print(f"- Total rows: {total_rows}")
            print(f"- Rows with position: {filled_rows} ({100 * filled_rows / total_rows:.1f}%)")
            print(f"- Total players: {total_players}")
            print(f"- Players with position: {players_with_position} ({100 * players_with_position / total_players:.1f}%)")
            print("\nPosition samples (player_id, position, position_depth, count):")
            for sample in position_samples:
                print(f"  {sample}")
                
            conn.close()
            return
        
        # Get depth chart data (either from API or test data)
        if args.test_mode:
            logging.info("Using test data")
            depth_data = create_manual_test_data()
        else:
            logging.info("Fetching data from API")
            depth_data = fetch_depth_charts()
        
        # If requested, just dump the response and exit
        if args.dump_response:
            file_path = '/Users/lukesmac/Models/nba/data/depth_charts_dump.json'
            with open(file_path, 'w') as f:
                json.dump(depth_data, f, indent=2)
            print(f"API response dumped to {file_path}")
            return
        
        # Save raw data only if requested
        if args.save_json:
            save_raw_depth_charts(depth_data)
            logging.info("Saved raw depth chart data to JSON file")
        else:
            logging.info("Skipping JSON file creation (use --save-json to enable)")
        
        # Parse player positions
        player_positions = parse_depth_charts(depth_data)
        
        if not player_positions:
            print("No player positions found in depth charts.")
            logging.error("Failed to parse any player positions from the data.")
            
            # For debugging, try to determine what's wrong with the input data
            if not args.test_mode:
                print("Trying with test data to verify parsing logic...")
                test_data = create_manual_test_data()
                test_positions = parse_depth_charts(test_data)
                if test_positions:
                    print(f"Test data parsing successful ({len(test_positions)} positions).")
                    print("The API response format might be different than expected.")
                    print("Run with --dump-response to examine the API output.")
                else:
                    print("Test data parsing also failed. The parsing logic may need revision.")
            
            return
        
        # Update database
        conn = connect_to_db(args.db_path)
        
        # If force flag is set, clear all existing positions first
        if args.force:
            print("Force flag set - clearing all existing positions...")
            cursor = conn.cursor()
            cursor.execute("UPDATE player_game_stats SET position = NULL")
            conn.commit()
        
        updated_count = update_player_positions(conn, player_positions)
        
        # Final verification
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM player_game_stats WHERE position IS NOT NULL")
        rows_with_position = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_game_stats WHERE position_depth IS NOT NULL")
        rows_with_position_depth = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_game_stats")
        total_rows = cursor.fetchone()[0]
        
        # Get some samples for display
        cursor.execute("""
        SELECT player_id, position, position_depth FROM player_game_stats 
        WHERE position IS NOT NULL AND position_depth IS NOT NULL
        GROUP BY player_id, position, position_depth
        ORDER BY position, position_depth
        LIMIT 15
        """)
        position_samples = cursor.fetchall()
        
        conn.close()
        
        print(f"Successfully updated positions for {updated_count} players in database.")
        print(f"Total positions parsed: {len(player_positions)}")
        print(f"Database stats:")
        print(f"- Rows with position: {rows_with_position}/{total_rows} ({100 * rows_with_position / total_rows:.1f}%)")
        print(f"- Rows with position_depth: {rows_with_position_depth}/{total_rows} ({100 * rows_with_position_depth / total_rows:.1f}%)")
        
        print("\nPosition samples (player_id, position, position_depth):")
        for sample in position_samples:
            print(f"  {sample}")
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"Error: {e}")

if __name__ == "__main__":
    main()