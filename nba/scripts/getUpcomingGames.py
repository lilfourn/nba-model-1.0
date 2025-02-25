import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from statistics import mean

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API configuration
API_CONFIG = {
    "schedule_url": "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate",
    "odds_url": "https://tank01-fantasy-stats.p.rapidapi.com/getNBABettingOdds",
    "headers": {
        "x-rapidapi-key": "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade",
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
}

def get_formatted_dates(num_days: int = 3) -> List[str]:
    """Get formatted dates for today and the next specified number of days."""
    dates = []
    current_date = datetime.now()
    
    for i in range(num_days):
        date = current_date + timedelta(days=i)
        formatted_date = date.strftime("%Y%m%d")
        dates.append(formatted_date)
    
    return dates

def calculate_average_odds(sportsbooks: List[Dict]) -> Dict[str, float]:
    """
    Calculate average moneyline odds from all sportsbooks.
    
    Args:
        sportsbooks: List of sportsbook data
        
    Returns:
        Dict with average home and away moneyline odds
    """
    home_odds = []
    away_odds = []
    
    for book in sportsbooks:
        if 'odds' in book:
            odds = book['odds']
            if 'homeTeamMLOdds' in odds and odds['homeTeamMLOdds']:
                try:
                    home_odds.append(float(odds['homeTeamMLOdds']))
                except ValueError:
                    pass
            if 'awayTeamMLOdds' in odds and odds['awayTeamMLOdds']:
                try:
                    away_odds.append(float(odds['awayTeamMLOdds']))
                except ValueError:
                    pass
    
    return {
        'avg_home_ml': round(mean(home_odds), 2) if home_odds else None,
        'avg_away_ml': round(mean(away_odds), 2) if away_odds else None
    }

def get_odds_for_date(date: str) -> Dict[str, Dict]:
    """
    Get betting odds for all games on a specific date.
    
    Args:
        date (str): Date in YYYYMMDD format
        
    Returns:
        Dict mapping game IDs to their odds
    """
    try:
        response = requests.get(
            API_CONFIG["odds_url"],
            headers=API_CONFIG["headers"],
            params={"gameDate": date, "itemFormat": "list"}
        )
        response.raise_for_status()
        data = response.json()
        
        odds_by_game = {}
        if data and 'body' in data:
            for game in data['body']:
                if 'gameID' in game and 'sportsBooks' in game:
                    odds_by_game[game['gameID']] = calculate_average_odds(game['sportsBooks'])
        
        return odds_by_game
    except Exception as e:
        logging.error(f"Error fetching odds for date {date}: {str(e)}")
        return {}

def get_games_for_date(date: str) -> Optional[Dict]:
    """Get NBA games scheduled for a specific date."""
    try:
        response = requests.get(
            API_CONFIG["schedule_url"],
            headers=API_CONFIG["headers"],
            params={"gameDate": date}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching games for date {date}: {str(e)}")
        return None

def get_upcoming_games() -> Dict[str, List[Dict]]:
    """
    Get NBA games and odds for today, tomorrow, and the day after.
    Only includes games that have odds data available.
    
    Returns:
        Dict[str, List[Dict]]: Dictionary mapping dates to lists of game data with odds
    """
    upcoming_games = {}
    dates = get_formatted_dates()
    
    for date in dates:
        games_data = get_games_for_date(date)
        odds_data = get_odds_for_date(date)
        
        if games_data and 'body' in games_data:
            games_list = []
            body_data = games_data['body']
            
            # Handle both dictionary and list responses
            if isinstance(body_data, dict):
                for game_id, game_info in body_data.items():
                    if isinstance(game_info, dict) and game_id in odds_data:
                        game_info['gameID'] = game_id
                        game_info.update(odds_data[game_id])
                        games_list.append(game_info)
            elif isinstance(body_data, list):
                for game in body_data:
                    if game['gameID'] in odds_data:
                        game.update(odds_data[game['gameID']])
                        games_list.append(game)
            
            upcoming_games[date] = games_list
            logging.info(f"Found {len(games_list)} games with odds for {date}")
        else:
            upcoming_games[date] = []
            logging.warning(f"No games found for {date}")
    
    return upcoming_games

def format_game_info(game: Dict) -> str:
    """Format game information into a readable string."""
    away_team = game.get('away', 'Unknown')
    home_team = game.get('home', 'Unknown')
    game_id = game.get('gameID', 'No ID')
    away_ml = game.get('avg_away_ml', 'No odds')
    home_ml = game.get('avg_home_ml', 'No odds')
    
    return f"{away_team} ({away_ml}) @ {home_team} ({home_ml}) - {game_id}"

if __name__ == "__main__":
    # Example usage
    games = get_upcoming_games()
    
    for date, games_list in games.items():
        formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
        print(f"\nGames for {formatted_date}:")
        if games_list:
            for game in games_list:
                print(format_game_info(game))
        else:
            print("No games with odds available")