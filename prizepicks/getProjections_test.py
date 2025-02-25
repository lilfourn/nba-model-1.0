import requests
import json
from datetime import datetime
import pprint

def test_api_response():
    """Get a sample of the API response to examine opponent information."""
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
        
        # Find an NBA projection to examine
        nba_projection = None
        for item in data.get('data', []):
            if item['type'] == 'projection':
                # Check if this is an NBA projection (league ID 7)
                league_data = item.get('relationships', {}).get('league', {}).get('data', {})
                if league_data.get('id') == '7':
                    nba_projection = item
                    break
        
        if not nba_projection:
            print("No NBA projections found")
            return
        
        # Examine the projection's relationships
        projection_relationships = nba_projection.get('relationships', {})
        projection_id = nba_projection['id']
        
        print(f"\nNBA Projection ID: {projection_id}")
        print("\nProjection Relationships:")
        pprint.pprint(projection_relationships)
        
        # Look for game relationship
        game_rel = projection_relationships.get('game', {}).get('data', {})
        if game_rel:
            game_id = game_rel.get('id')
            print(f"\nGame ID: {game_id}")
            
            # Find the game data in included
            game_data = None
            for item in data.get('included', []):
                if item['type'] == 'game' and item['id'] == game_id:
                    game_data = item
                    break
            
            if game_data:
                print("\nGame Data:")
                pprint.pprint(game_data)
            else:
                print("Game data not found in included items")
        
        # Also check player data for team information
        player_rel = projection_relationships.get('new_player', {}).get('data', {})
        if player_rel:
            player_id = player_rel.get('id')
            print(f"\nPlayer ID: {player_id}")
            
            # Find the player data in included
            player_data = None
            for item in data.get('included', []):
                if item['type'] == 'new_player' and item['id'] == player_id:
                    player_data = item
                    break
            
            if player_data:
                print("\nPlayer Data:")
                pprint.pprint(player_data)
            else:
                print("Player data not found in included items")
        
        # Save a sample of the full response for further examination
        with open('api_response_sample.json', 'w') as f:
            json.dump(data, f, indent=2)
            print("\nFull API response saved to api_response_sample.json")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching projections: {e}")

if __name__ == "__main__":
    test_api_response()