import json
import os

# Path to the teams response sample
sample_file = "teams_response_sample.json"

if os.path.exists(sample_file):
    with open(sample_file, 'r') as f:
        team_data = json.load(f)
    
    # Extract team name and abbreviation
    team_abv = team_data.get('teamAbv')
    team_city = team_data.get('teamCity')
    print(f"\nTeam: {team_city} {team_abv}")
    
    # Extract defensive stats
    defensive_stats = team_data.get('defensiveStats', {})
    if defensive_stats:
        print("\nDefensive Stats by Position:")
        print("=" * 30)
        
        # Print points allowed
        print("\nPoints Allowed:")
        print(f"  Home: {defensive_stats.get('ptsHome', 'N/A')}")
        print(f"  Away: {defensive_stats.get('ptsAway', 'N/A')}")
        
        # Print blocks by position
        blocks = defensive_stats.get('blk', {})
        if blocks:
            print("\nBlocks Allowed by Position:")
            for pos, value in blocks.items():
                if pos != 'Total':
                    print(f"  {pos}: {value}")
            print(f"  Total: {blocks.get('Total', 'N/A')}")
        
        # Print steals by position
        steals = defensive_stats.get('stl', {})
        if steals:
            print("\nSteals Allowed by Position:")
            for pos, value in steals.items():
                if pos != 'Total':
                    print(f"  {pos}: {value}")
            print(f"  Total: {steals.get('Total', 'N/A')}")
        
        # Print field goal attempts by position
        fga = defensive_stats.get('fga', {})
        if fga:
            print("\nFG Attempts Allowed by Position:")
            for pos, value in fga.items():
                if pos != 'Total':
                    print(f"  {pos}: {value}")
            print(f"  Total: {fga.get('Total', 'N/A')}")
        
        # Print field goal makes by position
        fgm = defensive_stats.get('fgm', {})
        if fgm:
            print("\nFG Makes Allowed by Position:")
            for pos, value in fgm.items():
                if pos != 'Total':
                    print(f"  {pos}: {value}")
            print(f"  Total: {fgm.get('Total', 'N/A')}")
        
        # Print 3-point makes by position
        tptfgm = defensive_stats.get('tptfgm', {})
        if tptfgm:
            print("\n3PT Makes Allowed by Position:")
            for pos, value in tptfgm.items():
                if pos != 'Total':
                    print(f"  {pos}: {value}")
            print(f"  Total: {tptfgm.get('Total', 'N/A')}")
        
        # Print assists by position
        ast = defensive_stats.get('ast', {})
        if ast:
            print("\nAssists Allowed by Position:")
            for pos, value in ast.items():
                if pos != 'Total':
                    print(f"  {pos}: {value}")
            print(f"  Total: {ast.get('Total', 'N/A')}")
    
    # Extract team stats if available
    team_stats = team_data.get('teamStats', {})
    if team_stats:
        print("\nTeam Stats:")
        print("=" * 30)
        # Print first 10 team stats
        count = 0
        for stat, value in team_stats.items():
            if count < 10:
                print(f"  {stat}: {value}")
                count += 1
    
    # Check for roster data
    roster = team_data.get('teamRoster', {})
    if roster:
        print("\nRoster Info:")
        print(f"  Number of players: {len(roster)}")
        # Get first player as example
        if len(roster) > 0:
            first_player_id = list(roster.keys())[0]
            first_player = roster[first_player_id]
            print("\nExample Player Data:")
            print(f"  Name: {first_player.get('longName', 'N/A')}")
            print(f"  Position: {first_player.get('pos', 'N/A')}")
            
            # Check if player has stats
            player_stats = first_player.get('stats', {})
            if player_stats:
                print("\nPlayer Stats:")
                for stat, value in list(player_stats.items())[:5]:
                    print(f"  {stat}: {value}")
    
    # Look for schedule info
    schedule = team_data.get('teamSchedule', {})
    if schedule:
        print("\nSchedule Info:")
        print(f"  Number of games: {len(schedule)}")
        
        # Get next 3 games
        next_games = list(schedule.items())[:3]
        print("\nUpcoming Games:")
        for game_id, game_info in next_games:
            print(f"  {game_info.get('gameDate', 'N/A')}: {game_info.get('away', 'N/A')} @ {game_info.get('home', 'N/A')}")
            print(f"    Game time: {game_info.get('gameTime', 'N/A')}")
    
else:
    print(f"File {sample_file} not found. Run the test_game_schedule_api.py script first.")