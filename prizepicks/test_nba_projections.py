import requests
import json
import pprint

def test_nba_projections():
    """Get NBA-specific projections to examine opponent information."""
    url = 'https://partner-api.prizepicks.com/projections'
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    # Add a filter parameter to request only NBA projections
    params = {
        'filter[league_id]': '7'  # NBA league ID is 7
    }
    
    try:
        print(f"Fetching NBA projections from {url}")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Save the NBA-specific response
        with open('nba_api_response.json', 'w') as f:
            json.dump(data, f, indent=2)
            print(f"Saved NBA API response to nba_api_response.json")
        
        # Find all games in included data
        games = {}
        for item in data.get('included', []):
            if item['type'] == 'game':
                games[item['id']] = item
        
        print(f"Found {len(games)} NBA games in the API response")
        
        # Check if there are any games
        if games:
            print("\nExample NBA game data:")
            game_id, game = next(iter(games.items()))
            pprint.pprint(game)
            
            # Check for opponent information
            if 'metadata' in game['attributes'] and 'game_info' in game['attributes']['metadata']:
                game_info = game['attributes']['metadata']['game_info']
                print("\nTeams information:")
                pprint.pprint(game_info.get('teams', {}))
        
        # Find projections with game relationships
        nba_projections_with_games = []
        for item in data.get('data', []):
            if item['type'] == 'projection':
                # Verify it's NBA
                league_data = item.get('relationships', {}).get('league', {}).get('data', {})
                if league_data.get('id') == '7':
                    game_rel = item.get('relationships', {}).get('game', {}).get('data', {})
                    if game_rel and game_rel.get('id'):
                        nba_projections_with_games.append(item)
        
        print(f"\nFound {len(nba_projections_with_games)} NBA projections with game relationships")
        
        # If we found any NBA projections with games, show a sample
        if nba_projections_with_games:
            print("\nExample NBA projection with game relationship:")
            projection = nba_projections_with_games[0]
            print(f"Projection ID: {projection['id']}")
            
            # Get player info
            player_rel = projection.get('relationships', {}).get('new_player', {}).get('data', {})
            player_data = None
            if player_rel:
                player_id = player_rel.get('id')
                for inc in data.get('included', []):
                    if inc['type'] == 'new_player' and inc['id'] == player_id:
                        player_data = inc
                        break
            
            if player_data:
                print(f"Player: {player_data['attributes'].get('name')}")
                print(f"Team: {player_data['attributes'].get('team')}")
            
            # Get game info
            game_rel = projection.get('relationships', {}).get('game', {}).get('data', {})
            if game_rel:
                game_id = game_rel.get('id')
                if game_id in games:
                    game = games[game_id]
                    print("\nGame attributes:")
                    pprint.pprint(game.get('attributes', {}))
                    
                    # Check game metadata for opponent info
                    if 'metadata' in game['attributes']:
                        metadata = game['attributes']['metadata']
                        if 'game_info' in metadata and 'teams' in metadata['game_info']:
                            teams = metadata['game_info']['teams']
                            print("\nTeams data:")
                            pprint.pprint(teams)
                            
                            # Match player team with home/away to find opponent
                            if player_data:
                                player_team = player_data['attributes'].get('team')
                                home_team = teams.get('home', {}).get('abbreviation')
                                away_team = teams.get('away', {}).get('abbreviation')
                                
                                print(f"\nPlayer's team: {player_team}")
                                print(f"Home team: {home_team}")
                                print(f"Away team: {away_team}")
                                
                                if player_team == home_team:
                                    print(f"Player is on home team, opponent is: {away_team}")
                                elif player_team == away_team:
                                    print(f"Player is on away team, opponent is: {home_team}")
        else:
            print("No NBA projections with game relationships found")
                
    except requests.exceptions.RequestException as e:
        print(f"Error fetching projections: {e}")

if __name__ == "__main__":
    test_nba_projections()