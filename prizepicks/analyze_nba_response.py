import json
import pprint

def analyze_nba_response():
    """Analyze the NBA API response to find alternative ways to get opponent information."""
    try:
        with open('nba_api_response.json', 'r') as f:
            data = json.load(f)
        
        # Check how many NBA projections we have
        nba_projections = []
        for item in data.get('data', []):
            if item['type'] == 'projection':
                league_data = item.get('relationships', {}).get('league', {}).get('data', {})
                if league_data and league_data.get('id') == '7':  # NBA league ID is 7
                    nba_projections.append(item)
        
        print(f"Found {len(nba_projections)} NBA projections in the response")
        
        # Look at a sample projection
        if nba_projections:
            print("\nSample NBA Projection:")
            sample_projection = nba_projections[0]
            pprint.pprint(sample_projection)
            
            # Get the player data for this projection
            player_rel = sample_projection.get('relationships', {}).get('new_player', {}).get('data', {})
            if player_rel:
                player_id = player_rel.get('id')
                player_data = None
                for item in data.get('included', []):
                    if item['type'] == 'new_player' and item['id'] == player_id:
                        player_data = item
                        break
                
                if player_data:
                    print("\nPlayer Data:")
                    pprint.pprint(player_data)
                    
                    # Check if the player attributes contain opponent info
                    player_attrs = player_data.get('attributes', {})
                    print("\nPlayer Attributes:")
                    for key, value in player_attrs.items():
                        print(f"{key}: {value}")
                    
                    # Check if there's a game in the response that matches this player's team
                    player_team = player_attrs.get('team')
                    if player_team:
                        print(f"\nLooking for games involving team: {player_team}")
                        
                        for item in data.get('included', []):
                            if item['type'] == 'game':
                                game_attrs = item.get('attributes', {})
                                metadata = game_attrs.get('metadata', {})
                                
                                if 'game_info' in metadata and 'teams' in metadata['game_info']:
                                    teams = metadata['game_info']['teams']
                                    home_team = teams.get('home', {}).get('abbreviation')
                                    away_team = teams.get('away', {}).get('abbreviation')
                                    
                                    if home_team == player_team or away_team == player_team:
                                        print("\nFound matching game:")
                                        pprint.pprint(item)
                                        print(f"Home team: {home_team}")
                                        print(f"Away team: {away_team}")
                                        
                                        if home_team == player_team:
                                            opponent = away_team
                                        else:
                                            opponent = home_team
                                            
                                        print(f"Opponent for {player_team} is: {opponent}")
                                        break
            
            # Look for any hint in the projection attributes
            print("\nProjection Attributes:")
            pprint.pprint(sample_projection.get('attributes', {}))
            
            # Check if there's any opponent info in other related objects
            for relationship_name, relationship in sample_projection.get('relationships', {}).items():
                if relationship.get('data'):
                    related_id = relationship['data'].get('id')
                    related_type = relationship['data'].get('type')
                    
                    if related_id and related_type:
                        print(f"\nChecking related {related_type} with ID {related_id}:")
                        
                        for item in data.get('included', []):
                            if item['type'] == related_type and item['id'] == related_id:
                                print(f"Found related {related_type}:")
                                pprint.pprint(item)
                                break
            
            # Check stat_average entities, as they might contain opponent info
            stat_avg_rel = sample_projection.get('relationships', {}).get('stat_average', {}).get('data', {})
            if stat_avg_rel:
                stat_avg_id = stat_avg_rel.get('id')
                
                if stat_avg_id:
                    for item in data.get('included', []):
                        if item['type'] == 'stat_average' and item['id'] == stat_avg_id:
                            print("\nStat Average Data:")
                            pprint.pprint(item)
                            
                            # Check if there's opponent info in attributes
                            if 'attributes' in item and 'opponent' in item['attributes']:
                                print(f"\nFound opponent in stat_average: {item['attributes']['opponent']}")
                            
                            break
        else:
            print("No NBA projections found in the response")
    
    except FileNotFoundError:
        print("NBA API response file not found. Run test_nba_projections.py first.")
    except json.JSONDecodeError:
        print("Invalid JSON in the NBA API response file.")

if __name__ == "__main__":
    analyze_nba_response()