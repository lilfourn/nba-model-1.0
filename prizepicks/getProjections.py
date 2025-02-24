import requests
import json
from datetime import datetime
import time

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

if __name__ == "__main__":
    projections = get_nba_projections()
    if projections and projections['projections']:
        sample = projections['projections'][0]
        print("\nSample projection with relationships:")
        print(f"Projection ID: {sample['id']}")
        if 'player' in sample:
            print(f"Player: {sample['player'].get('name', 'Unknown')}")
            print(f"Team: {sample['player'].get('team', 'Unknown')}")
        else:
            print("No player data found")
        print(f"Line Score: {sample['attributes'].get('line_score')}")
        print(f"Stat Type: {sample['attributes'].get('stat_type')}")