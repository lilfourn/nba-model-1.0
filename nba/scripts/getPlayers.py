import requests
import sqlite3

url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerList"

headers = {
	"x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
	"x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

data = response.json()
# Store complete player information
player_map = []
if 'body' in data:
    player_map = [
        {
            'playerID': player['playerID'],
            'name': player['longName'],
            'position': player['pos'],
            'team': player['team'],
            'teamID': player['teamID']
        }
        for player in data['body']
        if all(key in player for key in ['playerID', 'longName', 'pos', 'team', 'teamID'])
    ]

# Create SQLite database
db_name = 'nba_players.db'
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

# Create players table with all fields

command1 = """CREATE TABLE IF NOT EXISTS
players(playerID TEXT PRIMARY KEY, name TEXT NOT NULL, position TEXT NOT NULL, team TEXT NOT NULL, teamID TEXT NOT NULL)"""

cursor.execute(command1)

# Insert player data
insert_query = """INSERT OR REPLACE INTO players (playerID, name, position, team, teamID) VALUES(?, ?, ?, ?, ?)"""

try: 
    # Insert all players
    for player in player_map:
        cursor.execute(insert_query, 
        (
            player['playerID'],
            player['name'],
            player['position'],
            player['team'],
            player['teamID']
        ))
    
    # Commit the changes
    connection.commit()
    print(f"Successfully added {len(player_map)} players to the database")

    all_rows = cursor.fetchall()
    
except sqlite3.Error as e:
    print(f"An error occurred: {e}")

finally:
    # Always close the connection
    connection.close()