# CLAUDE.md - Guidelines for NBA Stats Prediction Project

## Project Overview
- Building machine learning models to predict NBA player stats based on historical data and game odds
- Current progress: Complete data pipeline and predictive modeling system
- Web-based dashboard for visualization of predictions vs. projections

## Development Commands
- Run NBA stats fetcher: `python nba/scripts/getPlayerStats.py`
- Run team roster fetcher: `python nba/scripts/getTeamRoster.py` 
- Run upcoming games fetcher: `python nba/scripts/getUpcomingGames.py`
- Run projections script: `python prizepicks/getProjections.py`
- Store projections: `python prizepicks/store_projections.py`
- Run projections scheduler: `python prizepicks/run_projections_scheduler.py`

## Prediction and Analysis Commands
- Train prediction models: `python nba/models/player_predictions.py`
- Predict for specific player: `python nba/models/player_predictions.py --predict "Player Name"`
- Show top fantasy players: `python nba/models/player_predictions.py --top-n 20`
- Show most confident predictions: `python nba/models/player_predictions.py --confident`
- Export top predictions to CSV: `python nba/models/player_predictions.py --export-csv /path/to/output.csv`
- Compare against projections: `python nba/models/player_predictions.py --compare-projections`
- Generate dashboard data: `bash nba/generate_dashboard.sh`

## Dashboard
- Dashboard location: `nba/dashboard.html`
- Dashboard data directory: `nba/dashboard_data/`
- To update dashboard data: `bash nba/generate_dashboard.sh`
- Dashboard features:
  - Over/under predictions with confidence scores
  - Player search functionality
  - Statistical visualizations
  - Model performance metrics

## Coding Approach
- Break down complex tasks into smaller steps with detailed explanations
- Provide complete code examples with comments explaining functionality
- Include basic Python concepts for learning purposes
- Suggest improvements while keeping code accessible for beginners
- Focus on data processing techniques specific to sports statistics
- Explain machine learning concepts in straightforward terms

## Next Development Steps
- Implement additional model types (neural networks, ensemble methods)
- Add player matchup analysis
- Incorporate injury data
- Track prediction accuracy over time
- Create automated prediction deployment system