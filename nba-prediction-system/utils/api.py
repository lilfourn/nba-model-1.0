
import requests
import logging
from config.api_config import NBA_API_KEY, NBA_API_HOST

def make_nba_api_request(endpoint, params=None):
    """
    Make a request to the NBA API.
    
    Args:
        endpoint (str): API endpoint path
        params (dict): Query parameters
        
    Returns:
        dict: JSON response data or None if error
    """
    url = f"https://{NBA_API_HOST}/{endpoint}"
    
    headers = {
        "x-rapidapi-key": NBA_API_KEY,
        "x-rapidapi-host": NBA_API_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request error: {e}")
        return None
