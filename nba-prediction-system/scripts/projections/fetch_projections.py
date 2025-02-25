import requests
import json
from datetime import datetime
import time
import sqlite3
import schedule
import threading

def create_database():
    """Create SQLite database and tables if they don't exist."""
    conn = sqlite3.connect('current_projections.db')
    cursor = conn.cursor()
    
    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projections'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create new projections table with opponent column
        cursor.execute('''
        CREATE TABLE projections (
            id TEXT PRIMARY KEY,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            line_score REAL,
            stat_type TEXT,
            timestamp DATETIME,
            raw_data JSON
        )
        ''')
    else:
        # Check if opponent column exists
        cursor.execute("PRAGMA table_info(projections)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        # Add opponent column if it doesn't exist
        if 'opponent' not in column_names:
            cursor.execute("ALTER TABLE projections ADD COLUMN opponent TEXT")
    
    conn.commit()
    conn.close()

def store_projections(projections_data):
    """Store projections in SQLite database."""
    if not projections_data or not projections_data['projections']:
        return
        
    conn = sqlite3.connect('current_projections.db')
    cursor = conn.cursor()
    
    # First, clear existing projections as we want current data only
    cursor.execute('DELETE FROM projections')
    
    # Insert new projections
    timestamp = datetime.now().isoformat()
    for proj in projections_data['projections']:
        # Extract opponent from description field (contains opponent team abbreviation)
        opponent = proj['attributes'].get('description')
        
        cursor.execute('''
        INSERT INTO projections (id, player_name, team, opponent, line_score, stat_type, timestamp, raw_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            proj['id'],
            proj.get('player', {}).get('name'),
            proj.get('player', {}).get('team'),
            opponent,  # New opponent field
            proj['attributes'].get('line_score'),
            proj['attributes'].get('stat_type'),
            timestamp,
            json.dumps(proj)
        ))
    
    conn.commit()
    conn.close()
    print(f"Stored {len(projections_data['projections'])} projections in database")

def get_nba_projections():
    """Get NBA projections from PrizePicks API."""
    url = 'https://partner-api.prizepicks.com/projections'
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        print(f"Fetching projections from {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Organize data by type for easier lookup
        organized_data = {
            'projections': [],  # List of projections with player info embedded
            'new_players': {},  # Map of player ID to player data
            'stat_types': {},   # Map of stat type ID to stat type data
            'stat_averages': {},  # Map of stat average ID to stat average data
            'games': {}         # Map of game ID to game data
        }
        
        # First pass: collect all included data
        for item in data.get('included', []):
            item_type = item['type']
            item_id = item['id']
            
            if item_type == 'new_player':
                organized_data['new_players'][item_id] = item
            elif item_type == 'stat_type':
                organized_data['stat_types'][item_id] = item
            elif item_type == 'stat_average':
                organized_data['stat_averages'][item_id] = item
            elif item_type == 'game':
                organized_data['games'][item_id] = item
        
        # Second pass: process main projections and link relationships
        for item in data.get('data', []):
            if item['type'] == 'projection':
                # Check if this is an NBA projection (league ID 7)
                league_data = item.get('relationships', {}).get('league', {}).get('data', {})
                if league_data.get('id') != '7':
                    continue
                
                projection = {
                    'id': item['id'],
                    'attributes': item['attributes'],
                    'relationships': item.get('relationships', {}),
                }
                
                # Link player data
                if 'new_player' in item['relationships']:
                    player_rel = item['relationships']['new_player']['data']
                    if player_rel and player_rel['id'] in organized_data['new_players']:
                        player_data = organized_data['new_players'][player_rel['id']]
                        projection['player'] = player_data['attributes']
                
                organized_data['projections'].append(projection)
        
        print(f"Found {len(organized_data['projections'])} NBA projections")
        print(f"Found {len(organized_data['new_players'])} players")
        print(f"Found {len(organized_data['stat_types'])} stat types")
        
        return organized_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching projections: {e}")
        return None

def update_projections():
    """Function to be called by scheduler to update projections."""
    print(f"\n[{datetime.now()}] Updating projections...")
    projections = get_nba_projections()
    if projections:
        store_projections(projections)
    else:
        print("Failed to fetch projections")

def run_scheduler():
    """Run the scheduler in a separate thread."""
    schedule.every(5).minutes.do(update_projections)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    create_database()
    print("Database created/verified")
    
    # Run initial update
    update_projections()
    
    # Display current database status
    conn = sqlite3.connect('current_projections.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM projections")
    count = cursor.fetchone()[0]
    print(f"Database now contains {count} projections")
    
    # Show sample projections with opponent
    cursor.execute("SELECT player_name, team, opponent, stat_type, line_score FROM projections LIMIT 5")
    samples = cursor.fetchall()
    print("\nSample projections with opponent information:")
    for sample in samples:
        print(f"Player: {sample[0]}, Team: {sample[1]}, Opponent: {sample[2]}, Stat: {sample[3]}, Line: {sample[4]}")
    conn.close()
    
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")