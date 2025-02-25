import json
import pprint

def examine_games():
    """Examine game data in the API response to find opponent information."""
    try:
        with open('api_response_sample.json', 'r') as f:
            data = json.load(f)
        
        # Find all games in included data
        games = {}
        for item in data.get('included', []):
            if item['type'] == 'game':
                games[item['id']] = item
        
        print(f"Found {len(games)} games in the API response:")
        for game_id, game in games.items():
            print(f"\nGame ID: {game_id}")
            pprint.pprint(game)
        
        # Find projections with game relationships
        count = 0
        for item in data.get('data', []):
            if item['type'] == 'projection':
                game_rel = item.get('relationships', {}).get('game', {}).get('data', {})
                if game_rel and game_rel.get('id'):
                    count += 1
                    if count <= 3:  # Just show first 3 examples
                        print(f"\nProjection with game relationship:")
                        print(f"Projection ID: {item['id']}")
                        print(f"Game ID: {game_rel.get('id')}")
                        
                        # Find player info
                        player_rel = item.get('relationships', {}).get('new_player', {}).get('data', {})
                        if player_rel:
                            player_id = player_rel.get('id')
                            for inc in data.get('included', []):
                                if inc['type'] == 'new_player' and inc['id'] == player_id:
                                    print(f"Player: {inc['attributes'].get('name')}")
                                    print(f"Team: {inc['attributes'].get('team')}")
                                    break
                        
                        # Find game info if exists
                        game_id = game_rel.get('id')
                        if game_id in games:
                            game = games[game_id]
                            print("Game attributes:")
                            pprint.pprint(game.get('attributes', {}))
        
        print(f"\nTotal projections with game relationships: {count}")
                
    except FileNotFoundError:
        print("API response sample file not found. Run getProjections_test.py first.")
    except json.JSONDecodeError:
        print("Invalid JSON in the API response sample file.")

if __name__ == "__main__":
    examine_games()