# CLAUDE.md - Guidelines for NBA Stats Prediction Project

## Project Overview
- Building a neural network to predict NBA player stats based on historical data and game odds
- Current progress: API data collection pipeline for NBA projections
- Database storage of historical projections for future model training

## Development Commands
- Run NBA stats fetcher: `python nba/scripts/getPlayerStats.py`
- Run team roster fetcher: `python nba/scripts/getTeamRoster.py` 
- Run upcoming games fetcher: `python nba/scripts/getUpcomingGames.py`
- Run projections script: `python prizepicks/getProjections.py`
- Store projections: `python prizepicks/store_projections.py`
- Run projections scheduler: `python prizepicks/run_projections_scheduler.py`

## Coding Approach
- Break down complex tasks into smaller steps with detailed explanations
- Provide complete code examples with comments explaining functionality
- Include basic Python concepts for learning purposes
- Suggest improvements while keeping code accessible for beginners
- Focus on data processing techniques specific to sports statistics
- Explain neural network concepts in straightforward terms

## Next Development Steps
- Data preprocessing and feature engineering
- Neural network model architecture and implementation
- Training/validation pipeline setup
- Model evaluation and refinement
- Prediction visualization and analysis