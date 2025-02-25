import requests
import json
from datetime import datetime, timedelta

def get_nba_games_for_date(game_date):
    """Get NBA games for a specific date from the Tank01 API."""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate"
    
    # Format date as YYYYMMDD
    if isinstance(game_date, datetime):
        formatted_date = game_date.strftime("%Y%m%d")
    else:
        formatted_date = game_date
    
    querystring = {"gameDate": formatted_date}
    
    headers = {
        "x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching games for date {formatted_date}: {e}")
        return None

def get_nba_teams():
    """Get NBA teams data from the Tank01 API."""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeams"
    
    querystring = {
        "schedules": "true",
        "rosters": "true",
        "statsToGet": "averages",
        "topPerformers": "true",
        "teamStats": "true"
    }
    
    headers = {
        "x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NBA teams data: {e}")
        return None

def get_upcoming_week_games():
    """Get NBA games for the upcoming week."""
    today = datetime.now()
    games_by_date = {}
    
    # Get games for next 7 days
    for i in range(7):
        date = today + timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        print(f"Fetching games for {date_str}...")
        games_data = get_nba_games_for_date(date_str)
        
        if games_data and "body" in games_data:
            games_by_date[date_str] = games_data["body"]
            game_count = len(games_data["body"]) if games_data["body"] else 0
            print(f"Found {game_count} games for {date_str}")
        else:
            print(f"No games found or API error for {date_str}")
    
    return games_by_date

# Test the game schedule API for today's date
today = datetime.now().strftime("%Y%m%d")
print(f"Testing game schedule API for date: {today}")
games_data = get_nba_games_for_date(today)

# Save the response to a file for examination
with open("game_schedule_response.json", "w") as f:
    json.dump(games_data, f, indent=2)
print(f"Saved game schedule response to game_schedule_response.json")

# Test the teams API
print("\nTesting teams API...")
teams_data = get_nba_teams()

# Save a subset of the teams response to a file (full response may be very large)
if teams_data and "body" in teams_data:
    # Save just the first team as a sample
    sample_team = teams_data["body"][0] if teams_data["body"] else {}
    with open("teams_response_sample.json", "w") as f:
        json.dump(sample_team, f, indent=2)
    print(f"Saved first team data to teams_response_sample.json")
    
    # Print total number of teams
    team_count = len(teams_data["body"]) if teams_data["body"] else 0
    print(f"Found data for {team_count} NBA teams")
else:
    print("No teams data found or API error")