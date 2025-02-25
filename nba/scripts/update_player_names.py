#!/usr/bin/env python
"""
Update player names in player_game_stats table by matching playerIDs with nba_players.db

This script connects the playerIDs in the player_game_stats table with the 
names from nba_players.db and adds the names as a new column to the player_game_stats table.
"""

import sqlite3
import os

def main():
    # Define database paths
    player_game_stats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          "data", "player_game_stats.db")
    nba_players_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                   "nba_players.db")
    
    print(f"Player game stats DB: {player_game_stats_path}")
    print(f"NBA players DB: {nba_players_path}")
    
    # Connect to player_game_stats.db
    conn_stats = sqlite3.connect(player_game_stats_path)
    c_stats = conn_stats.cursor()
    
    # Check if name column exists, add it if it doesn't
    c_stats.execute("PRAGMA table_info(player_game_stats)")
    columns = [column[1] for column in c_stats.fetchall()]
    
    if "name" not in columns:
        print("Adding name column to player_game_stats table...")
        c_stats.execute("ALTER TABLE player_game_stats ADD COLUMN name TEXT")
    else:
        print("Name column already exists in player_game_stats table")
    
    # Connect to nba_players.db
    conn_players = sqlite3.connect(nba_players_path)
    c_players = conn_players.cursor()
    
    # Get all player IDs from player_game_stats
    c_stats.execute("SELECT DISTINCT player_id FROM player_game_stats WHERE name IS NULL")
    player_ids = c_stats.fetchall()
    
    print(f"Found {len(player_ids)} unique players without names in player_game_stats")
    
    # Match player IDs with names from nba_players.db
    matched_count = 0
    for player_id in player_ids:
        player_id = player_id[0]
        c_players.execute("SELECT name FROM players WHERE playerID = ?", (player_id,))
        result = c_players.fetchone()
        
        if result:
            player_name = result[0]
            c_stats.execute("UPDATE player_game_stats SET name = ? WHERE player_id = ?", 
                          (player_name, player_id))
            matched_count += 1
    
    print(f"Matched {matched_count} out of {len(player_ids)} player IDs with names")
    
    # Commit changes and close connections
    conn_stats.commit()
    print("Changes committed to database")
    
    # Get and print stats on matched vs unmatched records
    c_stats.execute("SELECT COUNT(*) FROM player_game_stats")
    total_records = c_stats.fetchone()[0]
    
    c_stats.execute("SELECT COUNT(*) FROM player_game_stats WHERE name IS NOT NULL")
    matched_records = c_stats.fetchone()[0]
    
    print(f"\nSummary:")
    print(f"Total records in player_game_stats: {total_records}")
    print(f"Records with matched names: {matched_records} ({matched_records/total_records*100:.1f}%)")
    print(f"Records without matched names: {total_records - matched_records} ({(total_records - matched_records)/total_records*100:.1f}%)")
    
    # Close connections
    conn_stats.close()
    conn_players.close()

if __name__ == "__main__":
    main()