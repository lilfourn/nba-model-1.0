import requests
import sqlite3
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API configuration
API_CONFIG = {
    "url": "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster",
    "headers": {
        "x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
}

def create_database():
    """Create SQLite database and tables for team rosters."""
    conn = sqlite3.connect('nba_rosters.db')
    cursor = conn.cursor()
    
    # Create team rosters table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS team_rosters (
        team_abv TEXT,
        player_id TEXT,
        player_name TEXT,
        jersey_num TEXT,
        position TEXT,
        PRIMARY KEY (team_abv, player_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    logging.info("Database and tables created successfully")

def get_team_roster(team_abv: str) -> List[Dict]:
    """
    Fetch roster data for a specific team.
    
    Args:
        team_abv: Team abbreviation (e.g., 'SAC' for Sacramento Kings)
        
    Returns:
        List of player dictionaries containing roster information
    """
    try:
        response = requests.get(
            API_CONFIG["url"],
            headers=API_CONFIG["headers"],
            params={"teamAbv": team_abv}
        )
        response.raise_for_status()
        data = response.json()
        
        if data["statusCode"] == 200 and "body" in data and "roster" in data["body"]:
            return data["body"]["roster"]
        else:
            logging.error(f"No roster data found for team {team_abv}")
            return []
            
    except Exception as e:
        logging.error(f"Error fetching roster for team {team_abv}: {str(e)}")
        return []

def store_roster_data(team_abv: str, roster_data: List[Dict]):
    """
    Store roster data in SQLite database.
    
    Args:
        team_abv: Team abbreviation
        roster_data: List of player dictionaries
    """
    conn = sqlite3.connect('nba_rosters.db')
    cursor = conn.cursor()
    
    try:
        # Clear existing team data
        cursor.execute("DELETE FROM team_rosters WHERE team_abv = ?", (team_abv,))
        
        # Insert new roster data
        for player in roster_data:
            cursor.execute('''
            INSERT INTO team_rosters (team_abv, player_id, player_name, jersey_num, position)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                team_abv,
                player["playerID"],
                player["longName"],
                player["jerseyNum"],
                player["pos"]
            ))
        
        conn.commit()
        logging.info(f"Successfully stored roster data for {team_abv}")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error storing roster data for team {team_abv}: {str(e)}")
    
    finally:
        conn.close()

def get_all_team_rosters(team_abvs: List[str]):
    """
    Fetch and store roster data for multiple teams.
    
    Args:
        team_abvs: List of team abbreviations
    """
    create_database()
    
    for team_abv in team_abvs:
        logging.info(f"Fetching roster for {team_abv}")
        roster_data = get_team_roster(team_abv)
        if roster_data:
            store_roster_data(team_abv, roster_data)

if __name__ == "__main__":
    # Team abbreviations from the existing nba_players.db database
    teams = [
        "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
        "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
        "OKC", "ORL", "PHI", "PHO", "POR", "SA", "SAC", "TOR", "UTA", "WAS"
    ]
    
    get_all_team_rosters(teams)