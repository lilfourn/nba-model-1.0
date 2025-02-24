import sqlite3
import json
import os
from getProjections import get_nba_projections
from datetime import datetime

# Define database paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTIONS_DB = os.path.join(SCRIPT_DIR, 'data', 'nba_projections.db')
NBA_PLAYERS_DB = '/Users/lukesmac/Models/nba_players.db'

def get_nba_player_id(projection_name, nba_conn):
    """Get NBA player ID for a given projection player name."""
    cursor = nba_conn.cursor()
    
    # Try exact match first
    cursor.execute("SELECT playerID FROM players WHERE name = ?", (projection_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    
    # Try case-insensitive match
    cursor.execute("SELECT playerID FROM players WHERE LOWER(name) = LOWER(?)", (projection_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    
    # Try removing periods and apostrophes
    cleaned_name = projection_name.replace(".", "").replace("'", "")
    cursor.execute("SELECT playerID FROM players WHERE LOWER(REPLACE(REPLACE(name, '.', ''), '''', '')) = LOWER(?)", (cleaned_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    
    return None

def create_database():
    """Create the database and tables if they don't exist."""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(PROJECTIONS_DB), exist_ok=True)
    
    conn = sqlite3.connect(PROJECTIONS_DB)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projections'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create new table with current schema
        cursor.execute("""
        CREATE TABLE projections (
            id TEXT,  
            playerID TEXT,
            name TEXT NOT NULL,
            stat_type TEXT NOT NULL,
            line_score REAL NOT NULL,
            team TEXT,
            start_time TEXT,
            created_at TEXT,
            projection_id TEXT  
        );
        """)
        # Create index on projection_id for faster lookups
        cursor.execute("CREATE INDEX idx_projection_id ON projections(projection_id);")
        print("Created new database with current schema")
    
    conn.commit()
    return conn

def is_combo_projection(player_name, stat_type):
    """Check if a projection is a combo projection.
    
    Args:
        player_name (str): Name of the player(s)
        stat_type (str): Type of stat being projected
    
    Returns:
        bool: True if this is a combo projection, False otherwise
    """
    # Check for multiple players in name (contains '+' or 'vs')
    if '+' in player_name or ' vs ' in player_name.lower():
        return True
    
    # Check for combo in stat type
    if '(Combo)' in stat_type:
        return True
    
    return False

def store_projections(data):
    """Store NBA projections in the database. Only adds new projections, never updates or removes existing ones."""
    if not data or not data.get('projections'):
        print("No projections to store")
        return
    
    conn = create_database()
    nba_conn = sqlite3.connect(NBA_PLAYERS_DB)
    
    cursor = conn.cursor()
    new_count = 0
    existing_count = 0
    matched_players = 0
    skipped_combos = 0
    
    current_time = datetime.now().isoformat()
    
    try:
        # Process each projection
        for projection in data['projections']:
            attributes = projection.get('attributes', {})
            player = projection.get('player', {})
            player_name = player.get('name', 'Unknown')
            stat_type = attributes.get('stat_type', '')
            
            # Skip combo projections
            if is_combo_projection(player_name, stat_type):
                skipped_combos += 1
                continue
            
            # Get NBA player ID
            player_id = None
            if player_name != 'Unknown':
                player_id = get_nba_player_id(player_name, nba_conn)
                if player_id:
                    matched_players += 1
            
            if attributes.get('line_score') is not None:
                try:
                    # Check if this exact projection already exists
                    cursor.execute("""
                    SELECT 1 FROM projections 
                    WHERE projection_id = ? 
                    AND name = ? 
                    AND stat_type = ? 
                    AND line_score = ? 
                    AND team = ? 
                    AND start_time = ?
                    LIMIT 1
                    """, (
                        projection['id'],
                        player_name,
                        stat_type,
                        attributes['line_score'],
                        player.get('team', ''),
                        attributes.get('start_time')
                    ))
                    exists = cursor.fetchone() is not None
                    
                    if exists:
                        existing_count += 1
                    else:
                        # Insert new projection
                        cursor.execute("""
                        INSERT INTO projections 
                        (id, playerID, name, stat_type, line_score, team, start_time, created_at, projection_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(datetime.now().timestamp()),  
                            player_id,
                            player_name,
                            stat_type,
                            attributes['line_score'],
                            player.get('team', ''),
                            attributes.get('start_time'),
                            current_time,
                            projection['id']  
                        ))
                        new_count += 1
                        
                except Exception as e:
                    print(f"Error storing projection {projection['id']}: {e}")
                    print(f"Player data: {player}")
                    print(f"Projection data: {attributes}")
        
        conn.commit()
        print(f"Successfully processed NBA projections:")
        print(f"- Added {new_count} new projections")
        print(f"- Found {existing_count} existing identical projections (preserved)")
        print(f"- Skipped {skipped_combos} combo projections")
        print(f"- Matched {matched_players} players with NBA player IDs")
        
        # Show total projections in database
        cursor.execute("SELECT COUNT(*) FROM projections")
        total_count = cursor.fetchone()[0]
        print(f"Total projections in database: {total_count}")
        
        # Show some sample data
        cursor.execute("""
            SELECT name, playerID, team, stat_type, line_score, start_time, created_at
            FROM projections 
            WHERE playerID IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        """)
        print("\nLatest projections with player IDs:")
        for row in cursor.fetchall():
            print(f"{row[0]} (ID: {row[1]}, Team: {row[2]}): {row[3]} = {row[4]} (Start: {row[5]}, Added: {row[6]})")
        
    except Exception as e:
        print(f"Error storing projections: {e}")
        conn.rollback()
    finally:
        conn.close()
        nba_conn.close()

if __name__ == "__main__":
    # Get projections
    projections = get_nba_projections()
    
    # Store them in database
    store_projections(projections)
