import sys
import os
sys.path.append('/Users/lukesmac/Models/nba-prediction-system')
from utils.api import make_nba_api_request
from utils.db import db_connection, save_df_to_db
from utils.logging import setup_logger
from config.paths import DATA_DIR

import pandas as pd
from typing import Dict, List

# Set up logging
logger = setup_logger("team_roster", "api")

# Database configuration
DB_PATH = DATA_DIR / 'raw/nba_rosters.db'

def create_database():
    """Create SQLite database and tables for team rosters."""
    with db_connection(DB_PATH) as conn:
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
    
    logger.info("Database and tables created successfully")

def get_team_roster(team_abv: str) -> List[Dict]:
    """
    Fetch roster data for a specific team.
    
    Args:
        team_abv: Team abbreviation (e.g., 'SAC' for Sacramento Kings)
        
    Returns:
        List of player dictionaries containing roster information
    """
    try:
        data = make_nba_api_request("getNBATeamRoster", {"teamAbv": team_abv})
        
        if data and data.get("statusCode") == 200 and "body" in data and "roster" in data["body"]:
            return data["body"]["roster"]
        else:
            logger.error(f"No roster data found for team {team_abv}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching roster for team {team_abv}: {str(e)}")
        return []

def store_roster_data(team_abv: str, roster_data: List[Dict]):
    """
    Store roster data in SQLite database.
    
    Args:
        team_abv: Team abbreviation
        roster_data: List of player dictionaries
    """
    with db_connection(DB_PATH) as conn:
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
            logger.info(f"Successfully stored roster data for {team_abv}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing roster data for team {team_abv}: {str(e)}")

def get_all_team_rosters(team_abvs: List[str]):
    """
    Fetch and store roster data for multiple teams.
    
    Args:
        team_abvs: List of team abbreviations
    """
    create_database()
    
    for team_abv in team_abvs:
        logger.info(f"Fetching roster for {team_abv}")
        roster_data = get_team_roster(team_abv)
        if roster_data:
            store_roster_data(team_abv, roster_data)

def main():
    """Main function to fetch and store NBA team rosters."""
    # Ensure data directory exists
    os.makedirs(DATA_DIR / 'raw', exist_ok=True)
    
    # Team abbreviations
    teams = [
        "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
        "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
        "OKC", "ORL", "PHI", "PHO", "POR", "SA", "SAC", "TOR", "UTA", "WAS"
    ]
    
    logger.info("Starting team roster fetching process")
    get_all_team_rosters(teams)
    logger.info("Team roster fetching process completed")

if __name__ == "__main__":
    main()