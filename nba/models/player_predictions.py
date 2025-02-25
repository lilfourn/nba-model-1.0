#!/usr/bin/env python
"""
Player Performance Prediction Model

This script trains linear regression models to predict NBA player performance metrics
using the cleaned data from our NBA stats database and compares against projections.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import logging
import requests
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
import argparse
import json
from datetime import datetime, timedelta

# Import enhance_features functions
try:
    from enhance_features import (
        get_nba_games_for_date, 
        get_nba_teams, 
        extract_defensive_stats_by_position,
        create_enhanced_features
    )
except ImportError:
    # If the module isn't in the path, provide simplified versions
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
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": API_HOST
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching games for date {formatted_date}: {e}")
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
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": API_HOST
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching NBA teams data: {e}")
            return None
            
    def extract_defensive_stats_by_position(teams_data):
        """Extract defensive stats by position for all teams."""
        defensive_stats = {}
        
        if teams_data and "body" in teams_data:
            for team_data in teams_data["body"]:
                team_abv = team_data.get('teamAbv')
                if not team_abv:
                    continue
                    
                # Get defensive stats
                team_def_stats = team_data.get('defensiveStats', {})
                if not team_def_stats:
                    continue
                    
                # Create a dictionary for this team
                defensive_stats[team_abv] = {
                    'points_home': float(team_def_stats.get('ptsHome', 0)),
                    'points_away': float(team_def_stats.get('ptsAway', 0)),
                    'position_defense': {}
                }
                
                # Extract position-specific defensive stats
                for stat_type in ['blk', 'stl', 'fga', 'fgm', 'tptfgm', 'ast']:
                    if stat_type not in team_def_stats:
                        continue
                        
                    # Get stats by position
                    for position in ['PG', 'SG', 'SF', 'PF', 'C']:
                        if position in team_def_stats[stat_type]:
                            # Initialize position data if not already there
                            if position not in defensive_stats[team_abv]['position_defense']:
                                defensive_stats[team_abv]['position_defense'][position] = {}
                            
                            # Add the specific stat
                            stat_value = float(team_def_stats[stat_type][position])
                            defensive_stats[team_abv]['position_defense'][position][stat_type] = stat_value
        
        return defensive_stats

# Configure logging
os.makedirs('/Users/lukesmac/Models/nba/logs', exist_ok=True)
logging.basicConfig(
    filename='/Users/lukesmac/Models/nba/logs/prediction_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database paths
CLEAN_DB_PATH = '/Users/lukesmac/Models/nba/data/clean_stats.db'
ENHANCED_DB_PATH = '/Users/lukesmac/Models/nba/data/enhanced_stats.db'
TRAIN_DB_PATH = '/Users/lukesmac/Models/nba/data/clean_stats_train.db'
TEST_DB_PATH = '/Users/lukesmac/Models/nba/data/clean_stats_test.db'
MODEL_DIR = '/Users/lukesmac/Models/nba/models/trained'
PROJECTIONS_DB_PATH = '/Users/lukesmac/Models/prizepicks/current_projections.db'

# API configuration for external data
API_KEY = "498668d019msh8d5c3dfa8440cd6p1a2b07jsn51e2b2f77ade"
API_HOST = "tank01-fantasy-stats.p.rapidapi.com"

# Target metrics to predict
TARGET_METRICS = [
    'points', 
    'rebounds', 
    'assists', 
    'fantasy_points', 
    'three_pointers_made',
    'steals',
    'blocks',
    'turnovers',
    'field_goals_made',
    'free_throws_made',
]

# Mapping between projections stat_type and our database column names
STAT_TYPE_MAPPING = {
    'Points': 'points',
    'Rebounds': 'rebounds',
    'Assists': 'assists',
    'Fantasy Score': 'fantasy_points',
    '3-PT Made': 'three_pointers_made',
    'Steals': 'steals',
    'Blocks': 'blocks',
    'Turnovers': 'turnovers',
    'FG Made': 'field_goals_made',
    'Free Throws Made': 'free_throws_made',
}

def load_projections(db_path=PROJECTIONS_DB_PATH):
    """Load projections from prizepicks database."""
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM projections"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Parse the JSON raw_data
        df['raw_data_parsed'] = df['raw_data'].apply(json.loads)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logging.info(f"Loaded {len(df)} projections from {db_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading projections from {db_path}: {str(e)}")
        return pd.DataFrame()

def load_data(db_path, is_train=True, use_enhanced=True):
    """
    Load data from SQLite database into a pandas DataFrame.
    
    Parameters:
    - db_path: Path to the database file
    - is_train: Whether this is training data (True) or test data (False)
    - use_enhanced: Whether to use enhanced features if available
    
    Returns:
    - DataFrame with player stats data
    """
    try:
        # Check if enhanced data is available and we want to use it
        if use_enhanced and os.path.exists(ENHANCED_DB_PATH):
            try:
                # Try loading from enhanced database
                conn = sqlite3.connect(ENHANCED_DB_PATH)
                query = "SELECT * FROM enhanced_player_stats"
                df = pd.read_sql_query(query, conn)
                conn.close()
                logging.info(f"Loaded enhanced data with {len(df)} records and {len(df.columns)} features")
            except Exception as e:
                logging.warning(f"Error loading enhanced data, falling back to clean data: {str(e)}")
                # Fall back to clean data
                conn = sqlite3.connect(db_path)
                query = "SELECT * FROM clean_player_stats"
                df = pd.read_sql_query(query, conn)
                conn.close()
        else:
            # Load from regular clean data
            conn = sqlite3.connect(db_path)
            query = "SELECT * FROM clean_player_stats"
            df = pd.read_sql_query(query, conn)
            conn.close()
        
        # Convert numeric columns stored as text back to float/int
        numeric_cols = [col for col in df.columns if col not in ['player_id', 'game_id', 'team', 'opponent', 'game_date', 'position', 'position_depth', 'name']]
        
        for col in numeric_cols:
            try:
                # Try converting to float first
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                # If that fails, leave as is
                pass
        
        # Sort chronologically
        df = df.sort_values(['player_id', 'game_date'])
        
        # Check if we need to create enhanced features
        if use_enhanced and 'points_trend_last_5' not in df.columns:
            logging.info("Enhanced features not found in data. Creating them now...")
            
            # Create a temporary enhanced database
            temp_enhanced_path = ENHANCED_DB_PATH
            
            # Generate enhanced features
            try:
                # Import function from enhance_features module or create locally
                from enhance_features import create_enhanced_features
                df = create_enhanced_features(db_path, temp_enhanced_path)
                logging.info(f"Created enhanced features successfully. Total features: {len(df.columns)}")
            except Exception as e:
                logging.error(f"Error creating enhanced features: {str(e)}")
                logging.info("Continuing with existing features")
        
        # Add trend features if using enhanced data but they're not present
        if use_enhanced and 'points_trend_last_5' not in df.columns:
            logging.info("Adding basic trend features...")
            
            # Group by player
            for player_id, player_df in df.groupby('player_id'):
                if len(player_df) >= 5:
                    player_idx = player_df.index
                    
                    # Calculate trend indicators for key stats using last 5 games
                    for stat in ['points', 'rebounds', 'assists', 'steals', 'blocks', 'fantasy_points']:
                        if stat in player_df.columns:
                            # Get last 5 games stats
                            last_5_games = player_df[stat].tail(5).values
                            
                            # Calculate slope using simple linear regression
                            x = np.arange(len(last_5_games))
                            slope, _ = np.polyfit(x, last_5_games, 1)
                            
                            # Normalize by the mean to get percentage trend per game
                            mean_value = np.mean(last_5_games)
                            if mean_value > 0:
                                normalized_slope = 100 * slope / mean_value
                            else:
                                normalized_slope = 0
                            
                            # Add trend indicator
                            df.loc[player_idx[-1], f'{stat}_trend_last_5'] = normalized_slope
        
        logging.info(f"Loaded {'training' if is_train else 'testing'} data with {len(df)} records and {len(df.columns)} features")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {db_path}: {str(e)}")
        raise

def prepare_features_and_targets(data, target_metrics=TARGET_METRICS):
    """
    Prepare features and targets for predictive modeling.
    We'll use various player stats and context features to predict future performance.
    """
    feature_data = data.copy()
    
    # Strip target values from feature set
    targets = {}
    for metric in target_metrics:
        if metric in feature_data.columns:
            targets[metric] = feature_data[metric].copy()
    
    # Remove columns we don't want to use as features
    cols_to_drop = [
        'id', 'game_id', 'prev_game_date', 'game_date',  # ID and date columns
        'team', 'opponent', 'name',                      # String identifiers  
    ] + target_metrics  # Remove current game's actual performance (target values)
    
    # Drop columns that exist in the dataframe
    cols_to_drop = [col for col in cols_to_drop if col in feature_data.columns]
    feature_data = feature_data.drop(columns=cols_to_drop)
    
    # Convert any remaining string columns to numeric if possible
    for col in feature_data.columns:
        if feature_data[col].dtype == 'object':
            try:
                feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce')
            except:
                # If conversion fails, drop the column
                feature_data = feature_data.drop(columns=[col])
                logging.info(f"Dropped non-numeric column: {col}")
    
    # Fill any remaining NaN values
    feature_data = feature_data.fillna(0)
    
    # Log feature info
    logging.info(f"Prepared {len(feature_data.columns)} features for prediction")
    
    return feature_data, targets

def train_models(X_train, targets, alpha=1.0, use_ensemble=True):
    """
    Train regression models for each target metric.
    
    Parameters:
    - X_train: DataFrame of features
    - targets: Dict of target Series
    - alpha: Regularization strength for Ridge regression
    - use_ensemble: Whether to use ensemble of models (Ridge, GBR, XGBoost)
    
    Returns:
    - Dict of trained models (contains ensembles if use_ensemble=True)
    """
    models = {}
    
    # Define historical performance weights for different models by stat type
    # These will be adjusted based on validation performance
    stat_model_weights = {
        'points': {'ridge': 0.25, 'gbr': 0.35, 'xgb': 0.40},
        'rebounds': {'ridge': 0.30, 'gbr': 0.30, 'xgb': 0.40},
        'assists': {'ridge': 0.20, 'gbr': 0.40, 'xgb': 0.40},
        'fantasy_points': {'ridge': 0.25, 'gbr': 0.35, 'xgb': 0.40},
        'three_pointers_made': {'ridge': 0.30, 'gbr': 0.30, 'xgb': 0.40},
        'steals': {'ridge': 0.20, 'gbr': 0.35, 'xgb': 0.45},
        'blocks': {'ridge': 0.20, 'gbr': 0.35, 'xgb': 0.45},
        'turnovers': {'ridge': 0.30, 'gbr': 0.35, 'xgb': 0.35},
        'field_goals_made': {'ridge': 0.30, 'gbr': 0.30, 'xgb': 0.40},
        'free_throws_made': {'ridge': 0.35, 'gbr': 0.30, 'xgb': 0.35},
    }
    
    # Fallback weights if a specific stat type isn't in the dictionary
    default_weights = {'ridge': 0.30, 'gbr': 0.30, 'xgb': 0.40}
    
    for metric, y_train in targets.items():
        # Skip if insufficient data
        if len(y_train) < 100:
            logging.warning(f"Insufficient data for {metric}, skipping model training")
            continue
        
        try:    
            if use_ensemble:
                # Train Ridge, GBR, and XGBoost models, then ensemble them
                logging.info(f"Training ensemble model for {metric}")
                
                # Ridge Regression pipeline
                ridge_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Ridge(alpha=alpha))
                ])
                
                # Gradient Boosting Regression pipeline with modest parameters
                gbr_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42
                    ))
                ])
                
                # XGBoost model with careful tuning to avoid overfitting
                xgb_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.01,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1  # Use all available cores
                    ))
                ])
                
                # Train all models
                ridge_model.fit(X_train, y_train)
                gbr_model.fit(X_train, y_train)
                xgb_model.fit(X_train, y_train)
                
                # Get the best weights for this metric based on historical performance
                weights = stat_model_weights.get(metric, default_weights)
                
                # Create a dictionary containing all models and metadata
                ensemble = {
                    'ridge': ridge_model,
                    'gbr': gbr_model,
                    'xgb': xgb_model,
                    'weights': weights,
                    'is_ensemble': True
                }
                
                models[metric] = ensemble
                logging.info(f"Trained ensemble model for {metric} with weights: {weights}")
                
            else:
                # Create a pipeline with scaling and ridge regression only
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Ridge(alpha=alpha))
                ])
                
                # Train the model
                model.fit(X_train, y_train)
                models[metric] = model
                logging.info(f"Trained Ridge model for {metric}")
                
        except Exception as e:
            logging.error(f"Error training model for {metric}: {str(e)}")
    
    return models

def predict_with_model(model, X):
    """
    Make a prediction with a model, handling ensemble models appropriately.
    
    Parameters:
    - model: A trained sklearn Pipeline or an ensemble dictionary
    - X: Feature data for prediction
    
    Returns:
    - Prediction array
    """
    if isinstance(model, dict) and model.get('is_ensemble', False):
        # This is an ensemble model
        ridge_pred = model['ridge'].predict(X)
        gbr_pred = model['gbr'].predict(X)
        
        # Add XGBoost predictions if available
        if 'xgb' in model:
            xgb_pred = model['xgb'].predict(X)
            
            # Weighted average of all three model predictions
            weights = model['weights']
            prediction = (
                weights['ridge'] * ridge_pred + 
                weights['gbr'] * gbr_pred + 
                weights['xgb'] * xgb_pred
            )
        else:
            # Fall back to just Ridge + GBR if XGBoost isn't available
            weights = model['weights']
            prediction = (weights['ridge'] * ridge_pred + weights['gbr'] * gbr_pred)
            
        return prediction
    else:
        # This is a regular sklearn Pipeline
        return model.predict(X)

def evaluate_models(models, X_test, y_test_dict):
    """
    Evaluate models on test data.
    
    Parameters:
    - models: Dict of trained models
    - X_test: Test features
    - y_test_dict: Dict of test target Series
    
    Returns:
    - DataFrame with evaluation metrics
    """
    results = []
    
    for metric, model in models.items():
        if metric not in y_test_dict:
            continue
            
        y_test = y_test_dict[metric]
        
        # Make predictions
        y_pred = predict_with_model(model, X_test)
        
        # Calculate error metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Percentage error relative to average value
        avg_value = y_test.mean()
        pct_error = mae / avg_value * 100 if avg_value > 0 else float('inf')
        
        # Determine model type
        if isinstance(model, dict) and model.get('is_ensemble', False):
            model_type = "Ensemble"
        else:
            model_type = type(model[-1]).__name__
        
        results.append({
            'Metric': metric,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Avg Value': avg_value,
            'Error %': pct_error,
            'Model': model_type
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by percentage error
    results_df = results_df.sort_values('Error %')
    
    return results_df

def save_models(models, dir_path=MODEL_DIR):
    """Save trained models to disk."""
    os.makedirs(dir_path, exist_ok=True)
    
    for metric, model in models.items():
        # Create a valid filename
        filename = f"{metric.replace(' ', '_')}_model.joblib"
        path = os.path.join(dir_path, filename)
        
        try:
            joblib.dump(model, path)
            logging.info(f"Saved {metric} model to {path}")
        except Exception as e:
            logging.error(f"Error saving {metric} model: {str(e)}")

def predict_player_performance(player_name, models, data, metric_errors=None, num_future_games=1, opponent=None):
    """
    Predict future performance for a specific player.
    
    Parameters:
    - player_name: Name of the player to predict
    - models: Dict of trained models
    - data: DataFrame containing player data
    - metric_errors: Dict mapping metrics to their error rates (for confidence calculation)
    - num_future_games: Number of future games to predict
    - opponent: Optional opponent team for the prediction (to use opponent-specific features)
    
    Returns:
    - DataFrame with predictions and confidence scores
    """
    # Filter data for the specific player
    player_data = data[data['name'] == player_name].copy()
    
    if len(player_data) == 0:
        logging.warning(f"No data found for player: {player_name}")
        return None
    
    # Sort by game date (ensure chronological order)
    player_data = player_data.sort_values('game_date')
    
    # Get the latest game data as the basis for prediction
    latest_game = player_data.iloc[-1].copy()
    
    # If opponent is provided, update opponent information
    if opponent:
        logging.info(f"Setting opponent to {opponent} for prediction")
        latest_game['opponent'] = opponent
        
        # Try to update opponent-specific features if they exist
        opponent_cols = [col for col in latest_game.index if 'opponent' in col.lower() or 'opp_' in col.lower()]
        
        # If we have historical games against this opponent, use average stats against them
        past_games_vs_opp = player_data[player_data['opponent'] == opponent]
        if len(past_games_vs_opp) > 0:
            logging.info(f"Found {len(past_games_vs_opp)} past games vs {opponent}, using historical averages")
            
            # Update key stats with averages against this opponent
            for stat in TARGET_METRICS:
                if f"{stat}_vs_opp_avg" in latest_game.index:
                    latest_game[f"{stat}_vs_opp_avg"] = past_games_vs_opp[stat].mean()
                    
                # Also update the ratio features if they exist
                if f"{stat}_vs_opp_ratio" in latest_game.index:
                    opp_avg = past_games_vs_opp[stat].mean()
                    player_avg = player_data[stat].mean()
                    if player_avg > 0:
                        latest_game[f"{stat}_vs_opp_ratio"] = opp_avg / player_avg
    
    # Check for enhanced trend features
    trend_cols = [col for col in data.columns if 'trend_last_5' in col]
    if trend_cols:
        logging.info(f"Using {len(trend_cols)} trend features for prediction")
    
    # Check for rest/schedule features
    schedule_cols = [col for col in data.columns if any(x in col for x in ['is_back_to_back', 'is_long_rest', 'days_rest'])]
    if schedule_cols:
        logging.info(f"Using {len(schedule_cols)} schedule/rest features for prediction")
    
    # Check for defensive stats
    defense_cols = [col for col in data.columns if 'opp_defense_vs_position' in col]
    if defense_cols:
        logging.info(f"Using {len(defense_cols)} opponent defensive features for prediction")
    
    # Prepare features for prediction
    X_pred, _ = prepare_features_and_targets(pd.DataFrame([latest_game]))
    
    # Make predictions for each target metric
    predictions = {}
    confidence_scores = {}
    confidence_factors = {}  # Detailed breakdown of confidence components
    model_weights = {}  # Store model weights for analysis
    
    for metric, model in models.items():
        if model is not None:
            try:
                # Predict for this metric
                pred_value = predict_with_model(model, X_pred)
                if isinstance(pred_value, np.ndarray):
                    pred_value = pred_value[0]
                predictions[metric] = pred_value
                
                # Store model weights if this is an ensemble
                if isinstance(model, dict) and model.get('is_ensemble', False):
                    # Store all model weights used in the ensemble
                    for model_type, weight in model['weights'].items():
                        model_weights[f"{metric}_{model_type}_weight"] = weight
                
                # Calculate confidence components
                if metric in player_data.columns:
                    # 1. Player consistency component
                    recent_games = player_data.tail(10)  # Last 10 games for consistency calculation
                    stat_std = recent_games[metric].std()
                    stat_mean = recent_games[metric].mean()
                    
                    # Calculate coefficient of variation (normalized std deviation)
                    if stat_mean > 0:
                        cv = stat_std / stat_mean
                        player_consistency = 1 / (1 + cv)  # Transform to 0-1 scale
                    else:
                        player_consistency = 0.5  # Default for zero mean
                        
                    # Additional consistency measure - % of games within 20% of average
                    if stat_mean > 0:
                        games_within_range = recent_games[
                            (recent_games[metric] >= 0.8 * stat_mean) & 
                            (recent_games[metric] <= 1.2 * stat_mean)
                        ]
                        pct_games_within_range = len(games_within_range) / len(recent_games)
                        consistency_score = (player_consistency + pct_games_within_range) / 2
                    else:
                        consistency_score = player_consistency
                    
                    # 2. Recent form factors
                    recent_form_factor = 0.5  # Neutral by default
                    trend_explanation = "Neutral trend"
                    trend_col = f"{metric}_trend_last_5"
                    
                    if trend_col in latest_game:
                        trend_value = latest_game[trend_col]
                        # If trending strongly in either direction, adjust confidence
                        if trend_value > 20:  
                            recent_form_factor = 0.7  # Trending up strongly
                            trend_explanation = "Strong upward trend (+20%+)"
                        elif trend_value > 10:
                            recent_form_factor = 0.6  # Trending up moderately
                            trend_explanation = "Moderate upward trend (+10-20%)"
                        elif trend_value < -20:
                            recent_form_factor = 0.3  # Trending down strongly
                            trend_explanation = "Strong downward trend (-20%+)"
                        elif trend_value < -10:
                            recent_form_factor = 0.4  # Trending down moderately
                            trend_explanation = "Moderate downward trend (-10-20%)"
                        else:
                            recent_form_factor = 0.55  # Stable trend, slightly more confidence
                            trend_explanation = "Stable recent performance"
                            
                        # Add form consistency - if last game was significantly different from average
                        last_game_value = player_data[metric].iloc[-1]
                        if stat_mean > 0:
                            last_game_ratio = last_game_value / stat_mean
                            if last_game_ratio > 1.5:  # Last game was >50% above average
                                recent_form_factor -= 0.1
                                trend_explanation += ", last game significantly above average"
                            elif last_game_ratio < 0.5:  # Last game was <50% of average
                                recent_form_factor -= 0.1
                                trend_explanation += ", last game significantly below average"
                    
                    # 3. Matchup advantage/disadvantage
                    matchup_factor = 0.5  # Neutral by default
                    matchup_explanation = "Neutral matchup"
                    defense_col = f"opp_defense_vs_position_{metric}"
                    
                    if defense_col in latest_game and latest_game[defense_col] > 0:
                        # Higher values mean opponent allows more of this stat to the position
                        # Compare to league average
                        defense_val = latest_game[defense_col]
                        league_avg = data[defense_col].mean()
                        
                        if league_avg > 0:
                            ratio = defense_val / league_avg
                            if ratio > 1.2:  # Very favorable matchup
                                matchup_factor = 0.8
                                matchup_explanation = f"Very favorable matchup (+{((ratio-1)*100):.1f}% vs avg)"
                            elif ratio > 1.1:  # Favorable matchup
                                matchup_factor = 0.7
                                matchup_explanation = f"Favorable matchup (+{((ratio-1)*100):.1f}% vs avg)"
                            elif ratio < 0.8:  # Very unfavorable matchup
                                matchup_factor = 0.2
                                matchup_explanation = f"Very tough matchup ({((ratio-1)*100):.1f}% vs avg)"
                            elif ratio < 0.9:  # Unfavorable matchup
                                matchup_factor = 0.3
                                matchup_explanation = f"Tough matchup ({((ratio-1)*100):.1f}% vs avg)"
                            
                    # 4. Schedule situation
                    schedule_factor = 0.5  # Neutral by default
                    schedule_explanation = "Normal rest"
                    
                    if 'is_back_to_back' in latest_game and latest_game['is_back_to_back'] == 1:
                        schedule_factor = 0.3  # Back-to-back games reduce confidence
                        schedule_explanation = "Back-to-back game (potential fatigue)"
                    elif 'is_3rd_game_in_4_nights' in latest_game and latest_game['is_3rd_game_in_4_nights'] == 1:
                        schedule_factor = 0.4  # 3 games in 4 nights reduces confidence
                        schedule_explanation = "3rd game in 4 nights (potential fatigue)"
                    elif 'is_long_rest' in latest_game and latest_game['is_long_rest'] == 1:
                        schedule_factor = 0.6  # Long rest increases confidence
                        schedule_explanation = "Long rest (4+ days)"
                    elif 'days_rest' in latest_game:
                        days_rest = latest_game['days_rest']
                        if days_rest >= 2 and days_rest < 4:
                            schedule_factor = 0.55  # Good rest
                            schedule_explanation = f"{days_rest} days rest (normal recovery)"
                        elif days_rest == 1:
                            schedule_factor = 0.45  # Minimal rest
                            schedule_explanation = "Only 1 day rest"
                    
                    # 5. Model accuracy component
                    model_accuracy = 0.5  # Default
                    model_accuracy_explanation = "Average model accuracy"
                    
                    if metric_errors and metric in metric_errors:
                        # Convert error percentage to accuracy score (0-1 scale)
                        error_pct = metric_errors[metric]
                        model_accuracy = max(0, 1 - (error_pct / 100))
                        
                        # Classify model accuracy
                        if model_accuracy > 0.85:
                            model_accuracy_explanation = f"High model accuracy ({(model_accuracy*100):.1f}%)"
                        elif model_accuracy > 0.75:
                            model_accuracy_explanation = f"Good model accuracy ({(model_accuracy*100):.1f}%)"
                        elif model_accuracy > 0.65:
                            model_accuracy_explanation = f"Average model accuracy ({(model_accuracy*100):.1f}%)"
                        else:
                            model_accuracy_explanation = f"Below average model accuracy ({(model_accuracy*100):.1f}%)"
                    
                    # 6. Sample size adjustment - more games = more confidence
                    sample_factor = min(0.1, len(player_data) / 100)  # Max 0.1 boost for 100+ games
                    
                    # 7. Position consistency factor - some positions have more consistent roles
                    position_factor = 0.5
                    position = latest_game.get('position', '')
                    if position in ['C', 'PF']:  # Big men often have more consistent production
                        position_factor = 0.55
                    
                    # Calculate overall confidence based on multiple factors with weights
                    confidence = (
                        0.35 * model_accuracy +       # Model accuracy: 35% weight
                        0.20 * consistency_score +    # Player consistency: 20% weight
                        0.15 * recent_form_factor +   # Recent form: 15% weight
                        0.15 * matchup_factor +       # Matchup advantage: 15% weight
                        0.10 * schedule_factor +      # Schedule situation: 10% weight
                        0.05 * position_factor        # Position factor: 5% weight
                    ) + sample_factor                 # Sample size adjustment
                    
                    # Store overall confidence
                    confidence_scores[f"{metric}_confidence"] = confidence
                    
                    # Store individual confidence factors for analysis
                    confidence_factors[f"{metric}_factors"] = {
                        "model_accuracy": {
                            "score": model_accuracy,
                            "weight": 0.35,
                            "explanation": model_accuracy_explanation
                        },
                        "player_consistency": {
                            "score": consistency_score,
                            "weight": 0.20,
                            "explanation": f"Consistency score: {(consistency_score*100):.1f}%"
                        },
                        "recent_form": {
                            "score": recent_form_factor,
                            "weight": 0.15,
                            "explanation": trend_explanation
                        },
                        "matchup": {
                            "score": matchup_factor,
                            "weight": 0.15,
                            "explanation": matchup_explanation
                        },
                        "schedule": {
                            "score": schedule_factor,
                            "weight": 0.10,
                            "explanation": schedule_explanation
                        },
                        "position": {
                            "score": position_factor,
                            "weight": 0.05,
                            "explanation": f"Position factor for {position}"
                        }
                    }
                    
                    # Store individual confidence factors in flat structure for DataFrame
                    confidence_scores[f"{metric}_model_accuracy"] = model_accuracy
                    confidence_scores[f"{metric}_player_consistency"] = consistency_score
                    confidence_scores[f"{metric}_recent_form"] = recent_form_factor
                    confidence_scores[f"{metric}_matchup"] = matchup_factor
                    confidence_scores[f"{metric}_schedule"] = schedule_factor
                    confidence_scores[f"{metric}_position_factor"] = position_factor
                    
                    # Store explanations for each factor
                    confidence_scores[f"{metric}_model_accuracy_explanation"] = model_accuracy_explanation
                    confidence_scores[f"{metric}_consistency_explanation"] = f"Consistency score: {(consistency_score*100):.1f}%"
                    confidence_scores[f"{metric}_form_explanation"] = trend_explanation
                    confidence_scores[f"{metric}_matchup_explanation"] = matchup_explanation
                    confidence_scores[f"{metric}_schedule_explanation"] = schedule_explanation
                    
            except Exception as e:
                logging.error(f"Error predicting {metric} for {player_name}: {str(e)}")
                predictions[metric] = None
                confidence_scores[f"{metric}_confidence"] = 0.0
    
    # Create a DataFrame for predictions
    combined_dict = {**predictions, **confidence_scores, **model_weights}
    pred_df = pd.DataFrame([combined_dict])
    pred_df['player_name'] = player_name
    
    # Add detailed confidence factors as JSON
    pred_df['confidence_factors_json'] = json.dumps(confidence_factors)
    
    # Add some context information about the player
    for col in ['position', 'team', 'minutes_last_10']:
        if col in latest_game:
            pred_df[col] = latest_game[col]
    
    # Add opponent info
    pred_df['opponent'] = opponent if opponent else (latest_game['opponent'] if 'opponent' in latest_game else None)
    
    # Add games in dataset (more games = better prediction)
    pred_df['games_in_dataset'] = len(player_data)
    
    # Add recent trend info
    for stat in TARGET_METRICS:
        trend_col = f"{stat}_trend_last_5"
        if trend_col in latest_game:
            pred_df[trend_col] = latest_game[trend_col]
    
    # Add schedule info
    for col in ['is_back_to_back', 'is_3rd_game_in_4_nights', 'is_long_rest', 'days_rest']:
        if col in latest_game:
            pred_df[col] = latest_game[col]
    
    # Reorder columns with player name first
    cols = ['player_name'] + [col for col in pred_df.columns if col != 'player_name']
    pred_df = pred_df[cols]
    
    return pred_df

def get_all_players(data):
    """Get a list of all players in the dataset."""
    return data['name'].unique()

def get_most_confident_predictions(data, models, metric, n=10, metric_errors=None):
    """
    Get the most confident predictions for a specific metric across all players.
    
    Parameters:
    - data: DataFrame containing player data
    - models: Dict of trained models
    - metric: The target metric to predict
    - n: Number of players to return
    - metric_errors: Dict mapping metrics to their error rates
    
    Returns:
    - DataFrame with top N most confident predictions
    """
    if metric not in models:
        logging.error(f"No model available for {metric}")
        return None
    
    # Get unique players
    players = data['name'].unique()
    
    # Generate predictions for all players
    all_preds = []
    for player in players:
        # Only make prediction if enough data is available
        player_data = data[data['name'] == player]
        if len(player_data) >= 5:  # Minimum 5 games for reliable prediction
            pred_df = predict_player_performance(player, {metric: models[metric]}, data, metric_errors)
            if pred_df is not None:
                all_preds.append(pred_df)
    
    if not all_preds:
        return None
    
    # Combine all predictions
    all_preds_df = pd.concat(all_preds)
    
    # If confidence score is available, sort by it
    confidence_col = f"{metric}_confidence"
    if confidence_col in all_preds_df.columns:
        all_preds_df = all_preds_df.sort_values(confidence_col, ascending=False)
    
    # Select relevant columns
    selected_columns = ['player_name', metric, confidence_col, 'position', 'team', 'games_in_dataset']
    selected_columns = [col for col in selected_columns if col in all_preds_df.columns]
    
    # Return top N rows
    return all_preds_df[selected_columns].head(n)

def compare_predictions_with_projections(data, models, projections, metric_errors=None):
    """
    Compare our model predictions with external projections to identify value plays.
    
    Parameters:
    - data: DataFrame with our player data
    - models: Dict of trained models
    - projections: DataFrame with external projections
    - metric_errors: Dict mapping metrics to error rates
    
    Returns:
    - DataFrame with comparison results and over/under recommendations
    """
    if projections.empty:
        logging.warning("No projections available for comparison")
        return pd.DataFrame()
    
    # Get unique players in our dataset
    our_players = set(data['name'].unique())
    
    results = []
    
    # Process each projection
    for _, proj in projections.iterrows():
        player_name = proj['player_name']
        stat_type = proj['stat_type']
        line_score = proj['line_score']
        
        # Skip if we don't have a mapping for this stat type
        if stat_type not in STAT_TYPE_MAPPING:
            continue
            
        # Get our column name for this stat type
        our_stat = STAT_TYPE_MAPPING[stat_type]
        
        # Skip if we don't have a model for this stat
        if our_stat not in models:
            continue
        
        # Look for player in our dataset (exact match first)
        if player_name in our_players:
            matched_name = player_name
        else:
            # Try fuzzy matching (find closest name)
            best_match = None
            best_ratio = 0
            
            # Simple matching algorithm using partial string matching
            for our_player in our_players:
                # Calculate ratio
                if player_name.lower() in our_player.lower() or our_player.lower() in player_name.lower():
                    if len(player_name) > best_ratio:  # Use length as simple matching metric
                        best_match = our_player
                        best_ratio = len(player_name)
            
            if best_match:
                matched_name = best_match
            else:
                # No match found for this player
                continue
        
        # Make our prediction
        pred_df = predict_player_performance(matched_name, {our_stat: models[our_stat]}, data, metric_errors)
        
        if pred_df is not None and our_stat in pred_df.columns:
            # Get our predicted value
            our_prediction = pred_df[our_stat].values[0]
            
            # Get confidence score
            confidence_col = f"{our_stat}_confidence"
            confidence = pred_df[confidence_col].values[0] if confidence_col in pred_df.columns else 0.5
            
            # Determine if we predict over or under
            if our_prediction > line_score:
                prediction = "OVER"
                edge = our_prediction - line_score
            else:
                prediction = "UNDER"
                edge = line_score - our_prediction
            
            # Calculate % difference
            if line_score > 0:
                edge_percent = (abs(our_prediction - line_score) / line_score) * 100
            else:
                edge_percent = 0
                
            # Adjust confidence based on edge size 
            # (more confident when our prediction is further from the line)
            if edge_percent > 15:  # Big difference
                edge_confidence = 0.9  # Very confident in the edge
            elif edge_percent > 10:
                edge_confidence = 0.8
            elif edge_percent > 5:
                edge_confidence = 0.7
            else:
                edge_confidence = 0.5  # Small edge, less confident
                
            # Combine model confidence with edge confidence
            # (weighted average, with model confidence weighted 2x)
            final_confidence = ((confidence * 2) + edge_confidence) / 3
            
            # Add to results
            results.append({
                'player_name': player_name,
                'matched_name': matched_name,
                'stat_type': stat_type,
                'line_score': line_score,
                'our_prediction': round(our_prediction, 2),
                'prediction': prediction,
                'edge': round(edge, 2),
                'edge_percent': round(edge_percent, 2),
                'confidence': round(final_confidence, 4),
                'team': proj['team'],
                'opponent': proj.get('opponent', 'N/A'),  # Add opponent information
                'position': pred_df['position'].values[0] if 'position' in pred_df.columns else None,
                'games_in_dataset': pred_df['games_in_dataset'].values[0] if 'games_in_dataset' in pred_df.columns else 0,
                'timestamp': proj['timestamp']
            })
    
    if not results:
        return pd.DataFrame()
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by confidence (descending)
    results_df = results_df.sort_values('confidence', ascending=False)
    
    return results_df

def get_top_predictions_across_all_stats(data, models, n=50, metric_errors=None, prediction_days=7):
    """
    Get the top N most confident predictions across all stats and players.
    
    Parameters:
    - data: DataFrame containing player data
    - models: Dict of trained models
    - n: Total number of predictions to return
    - metric_errors: Dict mapping metrics to their error rates
    - prediction_days: Number of days in advance we're predicting
    
    Returns:
    - DataFrame with top N most confident predictions across all metrics
    """
    # Get predictions for all metrics and all players
    all_predictions = []
    
    for metric in TARGET_METRICS:
        if metric not in models:
            continue
            
        # Get unique players
        players = data['name'].unique()
        
        # Generate predictions for all players for this metric
        for player in players:
            # Only make prediction if enough data is available
            player_data = data[data['name'] == player]
            if len(player_data) >= 5:  # Minimum 5 games for reliable prediction
                pred_df = predict_player_performance(player, {metric: models[metric]}, data, metric_errors)
                if pred_df is not None:
                    # Extract just what we need
                    confidence_col = f"{metric}_confidence"
                    if confidence_col in pred_df.columns and metric in pred_df.columns:
                        prediction_record = {
                            'player_name': player,
                            'metric': metric,
                            'predicted_value': pred_df[metric].values[0],
                            'confidence': pred_df[confidence_col].values[0],
                            'position': pred_df['position'].values[0] if 'position' in pred_df.columns else None,
                            'team': pred_df['team'].values[0] if 'team' in pred_df.columns else None,
                            'games_in_dataset': pred_df['games_in_dataset'].values[0] if 'games_in_dataset' in pred_df.columns else None,
                            'prediction_days_ahead': prediction_days
                        }
                        all_predictions.append(prediction_record)
    
    if not all_predictions:
        return None
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Sort by confidence (descending)
    predictions_df = predictions_df.sort_values('confidence', ascending=False)
    
    # Return top N predictions
    return predictions_df.head(n)

def main():
    """Main function to train and evaluate models."""
    parser = argparse.ArgumentParser(description="Train and evaluate player performance prediction models")
    # Database paths
    parser.add_argument('--train-db', type=str, default=TRAIN_DB_PATH, help='Path to training database')
    parser.add_argument('--test-db', type=str, default=TEST_DB_PATH, help='Path to test database')
    parser.add_argument('--enhanced-db', type=str, default=ENHANCED_DB_PATH, help='Path to enhanced database')
    parser.add_argument('--projections-db', type=str, default=PROJECTIONS_DB_PATH, help='Path to projections database')
    
    # Model training options
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularization strength for Ridge regression')
    parser.add_argument('--no-enhanced-features', action='store_true', help='Disable use of enhanced features')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable ensemble models (use Ridge only)')
    parser.add_argument('--xgboost-only', action='store_true', help='Use only XGBoost models (no Ridge or GBR)')
    parser.add_argument('--adjust-weights', action='store_true', help='Adjust ensemble weights based on validation performance')
    
    # Basic query options
    parser.add_argument('--list-players', action='store_true', help='List all available players')
    parser.add_argument('--predict', type=str, default=None, help='Predict for specific player name')
    parser.add_argument('--opponent', type=str, default=None, help='Specify opponent for prediction')
    
    # Advanced query options
    parser.add_argument('--top-n', type=int, default=None, help='Predict for top N players by fantasy points')
    parser.add_argument('--confident', action='store_true', help='Show most confident predictions by stat')
    parser.add_argument('--export-csv', type=str, default=None, 
                       help='Export top 50 most confident predictions across all stats to CSV file')
    parser.add_argument('--prediction-days', type=int, default=7,
                       help='Number of days in advance to predict (for CSV output)')
    
    # Projection comparison options
    parser.add_argument('--compare-projections', action='store_true',
                       help='Compare our predictions against projections in projections database')
    parser.add_argument('--export-projections-csv', type=str, default=None,
                       help='Export projection comparisons to CSV file')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                       help='Minimum confidence threshold for projections (0.0-1.0)')
    
    # Dashboard options
    parser.add_argument('--dashboard-data', type=str, default=None,
                       help='Generate JSON data for visualization dashboard')
    parser.add_argument('--detailed-confidence', action='store_true',
                       help='Include detailed confidence breakdown in output')
    
    # Feature engineering options
    parser.add_argument('--create-enhanced-features', action='store_true',
                       help='Create enhanced features database from clean stats')
    parser.add_argument('--update-external-data', action='store_true',
                       help='Update external data from API endpoints')
                       
    # Confidence and explanation options
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                        help='Only show predictions with confidence above this threshold (0.0-1.0)')
    parser.add_argument('--explain-confidence', action='store_true',
                        help='Show detailed explanation of confidence factors')
    parser.add_argument('--sort-by-confidence', action='store_true',
                        help='Sort results by confidence score (highest first)')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        # Feature engineering mode - create enhanced features if requested
        if args.create_enhanced_features:
            print("Creating enhanced features database...")
            try:
                # Import function from enhance_features module
                from enhance_features import create_enhanced_features
                enhanced_df = create_enhanced_features(args.train_db, args.enhanced_db)
                print(f"Enhanced features created successfully. Total records: {len(enhanced_df)}")
                
                # Summarize new features
                new_features = [
                    col for col in enhanced_df.columns 
                    if any(x in col for x in ['trend_last_5', 'is_back_to_back', 
                                             'is_long_rest', 'is_3rd_game_in_4_nights',
                                             'opp_defense_vs_position'])
                ]
                
                print(f"\nNew features added ({len(new_features)}):")
                for feature in new_features:
                    print(f"- {feature}")
                return
            except Exception as e:
                print(f"Error creating enhanced features: {e}")
                return
        
        # Update external data if requested
        if args.update_external_data:
            print("Updating external data from API endpoints...")
            try:
                # Get upcoming games
                print("Fetching upcoming games...")
                games = get_nba_games_for_date(datetime.now().strftime("%Y%m%d"))
                if games and "body" in games:
                    print(f"Found {len(games['body'])} games for today")
                    for game in games["body"]:
                        print(f"  {game.get('away', '?')} @ {game.get('home', '?')}")
                
                # Get teams data
                print("\nFetching team data...")
                teams_data = get_nba_teams()
                if teams_data and "body" in teams_data:
                    print(f"Found data for {len(teams_data['body'])} NBA teams")
                    
                    # Extract defensive stats by position
                    defensive_stats = extract_defensive_stats_by_position(teams_data)
                    print(f"Extracted defensive stats for {len(defensive_stats)} teams")
                    
                    # Save defensive stats to a file for reference
                    with open('team_defensive_stats.json', 'w') as f:
                        json.dump(defensive_stats, f, indent=2)
                    print("Saved defensive stats to team_defensive_stats.json")
                return
            except Exception as e:
                print(f"Error updating external data: {e}")
                return
        
        # Determine whether to use enhanced features
        use_enhanced = not args.no_enhanced_features
        
        # Load data
        print(f"Loading training data{' with enhanced features' if use_enhanced else ''}...")
        train_data = load_data(args.train_db, is_train=True, use_enhanced=use_enhanced)
        
        print(f"Loading test data{' with enhanced features' if use_enhanced else ''}...")
        test_data = load_data(args.test_db, is_train=False, use_enhanced=use_enhanced)
        
        # List players if requested
        if args.list_players:
            players = get_all_players(train_data)
            print(f"\nAvailable players ({len(players)}):")
            for i, player in enumerate(sorted(players)):
                print(f"{player}", end=', ' if (i + 1) % 5 != 0 else '\n')
            print("\n")
            return
        
        # Prepare features and targets
        print("Preparing features and targets...")
        X_train, y_train_dict = prepare_features_and_targets(train_data)
        X_test, y_test_dict = prepare_features_and_targets(test_data)
        
        # Show feature info
        print(f"Training with {X_train.shape[1]} features")
        
        # Check for enhanced features
        if use_enhanced:
            trend_cols = [col for col in train_data.columns if 'trend_last_5' in col]
            if trend_cols:
                print(f"Using {len(trend_cols)} trend features")
            
            schedule_cols = [col for col in train_data.columns if any(x in col for x in ['is_back_to_back', 'is_long_rest', 'days_rest'])]
            if schedule_cols:
                print(f"Using {len(schedule_cols)} schedule/rest features")
            
            defense_cols = [col for col in train_data.columns if 'opp_defense_vs_position' in col]
            if defense_cols:
                print(f"Using {len(defense_cols)} opponent defensive features")
                
        # Train models with configured approach
        use_ensemble = not args.no_ensemble
        ensemble_description = "Ridge only"
        
        if use_ensemble:
            if args.xgboost_only:
                ensemble_description = "XGBoost only"
            else:
                ensemble_description = "Full ensemble (Ridge + GBR + XGBoost)"
                
        print(f"Training {ensemble_description} models with alpha={args.alpha}...")
        models = train_models(X_train, y_train_dict, alpha=args.alpha, use_ensemble=use_ensemble)
        
        # Save models
        print("Saving models...")
        save_models(models)
        
        # Evaluate models
        print("Evaluating models...")
        results = evaluate_models(models, X_test, y_test_dict)
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print(results.to_string(index=False))
        
        # Extract error rates for confidence calculation
        metric_errors = {row['Metric']: row['Error %'] for _, row in results.iterrows()}
        
        # Combine datasets for predictions
        combined_data = pd.concat([train_data, test_data])
        
        # If --compare-projections is provided, compare our predictions against projections
        if args.compare_projections or args.export_projections_csv or args.dashboard_data:
            print(f"\nLoading projections from {args.projections_db}...")
            projections = load_projections(args.projections_db)
            
            if not projections.empty:
                print(f"Loaded {len(projections)} projections. Comparing with our models...")
                comparison_results = compare_predictions_with_projections(
                    combined_data, models, projections, metric_errors
                )
                
                if not comparison_results.empty:
                    # Apply confidence threshold if specified
                    min_confidence = args.min_confidence
                    if min_confidence > 0:
                        filtered_results = comparison_results[comparison_results['confidence'] >= min_confidence]
                        if len(filtered_results) < len(comparison_results):
                            print(f"Filtered from {len(comparison_results)} to {len(filtered_results)} projections with confidence >= {min_confidence}")
                        comparison_results = filtered_results
                    
                    # Format confidence as percentage for display
                    display_df = comparison_results.copy()
                    display_df['confidence_pct'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
                    
                    # Add a value rating column
                    display_df['value_rating'] = display_df.apply(
                        lambda x: '⭐⭐⭐' if x['confidence'] > 0.8 else 
                                ('⭐⭐' if x['confidence'] > 0.7 else 
                                 ('⭐' if x['confidence'] > 0.6 else '')), 
                        axis=1
                    )
                    
                    # Add a performance expectation
                    display_df['expectation'] = display_df.apply(
                        lambda x: f"Significantly {'OVER' if x['prediction'] == 'OVER' else 'UNDER'}" if x['edge_percent'] > 20 else
                                (f"Moderately {'OVER' if x['prediction'] == 'OVER' else 'UNDER'}" if x['edge_percent'] > 10 else
                                 f"Slightly {'OVER' if x['prediction'] == 'OVER' else 'UNDER'}"),
                        axis=1
                    )
                    
                    # Organize by confidence tier
                    high_confidence = display_df[display_df['confidence'] > 0.8].copy()
                    med_confidence = display_df[(display_df['confidence'] > 0.7) & (display_df['confidence'] <= 0.8)].copy()
                    low_confidence = display_df[(display_df['confidence'] > 0.6) & (display_df['confidence'] <= 0.7)].copy()
                    
                    # Print summary by confidence tier
                    print(f"\nProjection Comparisons Summary:")
                    print(f"  High Confidence (>80%): {len(high_confidence)} projections")
                    print(f"  Medium Confidence (70-80%): {len(med_confidence)} projections")
                    print(f"  Low Confidence (60-70%): {len(low_confidence)} projections")
                    print(f"  Below Threshold (<60%): {len(display_df) - len(high_confidence) - len(med_confidence) - len(low_confidence)} projections")
                    
                    # Print top confident projections for each tier
                    if not high_confidence.empty:
                        print("\n🌟 HIGH CONFIDENCE PICKS (>80%) 🌟")
                        display_cols = ['player_name', 'team', 'opponent', 'stat_type', 'line_score', 'our_prediction', 
                                      'prediction', 'edge', 'edge_percent', 'confidence_pct', 'value_rating']
                        print(high_confidence[display_cols].head(10).to_string(index=False))
                    
                    if not med_confidence.empty and not args.min_confidence > 0.7:
                        print("\n🔵 MEDIUM CONFIDENCE PICKS (70-80%) 🔵")
                        display_cols = ['player_name', 'team', 'opponent', 'stat_type', 'line_score', 'our_prediction', 
                                      'prediction', 'edge', 'edge_percent', 'confidence_pct']
                        print(med_confidence[display_cols].head(10).to_string(index=False))
                    
                    # Count for each stat type with HIGH confidence
                    if not high_confidence.empty:
                        stat_counts = high_confidence['stat_type'].value_counts()
                        print("\nHigh Confidence Picks by Stat Type:")
                        for stat, count in stat_counts.items():
                            print(f"  {stat}: {count}")
                            
                        # Count for OVER vs UNDER with HIGH confidence
                        over_under = high_confidence['prediction'].value_counts()
                        print("\nHigh Confidence Distribution:")
                        for pred, count in over_under.items():
                            print(f"  {pred}: {count}")
                    
                    # Export to CSV if requested
                    if args.export_projections_csv:
                        # Add a timestamp column for when the analysis was run
                        comparison_results['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
                        
                        # Save to CSV
                        csv_path = args.export_projections_csv
                        comparison_results.to_csv(csv_path, index=False)
                        print(f"Exported projection comparisons to {csv_path}")
                    
                    # Generate dashboard data if requested
                    if args.dashboard_data:
                        # Create a directory for dashboard data if it doesn't exist
                        dashboard_dir = args.dashboard_data
                        os.makedirs(dashboard_dir, exist_ok=True)
                        
                        # Save projection comparisons as JSON
                        json_path = os.path.join(dashboard_dir, 'projection_comparisons.json')
                        comparison_results.to_json(json_path, orient='records', date_format='iso')
                        print(f"Saved projection comparisons to {json_path}")
                        
                        # Save model evaluation results as JSON
                        eval_json_path = os.path.join(dashboard_dir, 'model_evaluation.json')
                        results.to_json(eval_json_path, orient='records')
                        print(f"Saved model evaluation to {eval_json_path}")
                        
                        # Generate top predictions for dashboard
                        top_preds = get_top_predictions_across_all_stats(
                            combined_data, models, n=100, metric_errors=metric_errors
                        )
                        if top_preds is not None:
                            top_preds_json_path = os.path.join(dashboard_dir, 'top_predictions.json')
                            top_preds.to_json(top_preds_json_path, orient='records')
                            print(f"Saved top predictions to {top_preds_json_path}")
                        
                        # Generate data about players
                        player_data = combined_data[['name', 'position', 'team']].drop_duplicates()
                        player_games = combined_data.groupby('name').size().to_frame('games_played')
                        player_stats = combined_data.groupby('name')[TARGET_METRICS].mean().round(2)
                        player_info = player_games.join(player_stats)
                        player_info = player_data.set_index('name').join(player_info).reset_index()
                        
                        player_json_path = os.path.join(dashboard_dir, 'player_info.json')
                        player_info.to_json(player_json_path, orient='records')
                        print(f"Saved player info to {player_json_path}")
                        
                        print(f"All dashboard data generated in {dashboard_dir}")
                else:
                    print("No comparison results available.")
            else:
                print("No projections found in the database.")
        
        # If --export-csv is provided, generate and export top 50 predictions across all stats
        if args.export_csv:
            print(f"\nGenerating top 50 most confident predictions across all stats...")
            prediction_days = args.prediction_days
            
            top_predictions = get_top_predictions_across_all_stats(
                combined_data, models, n=50, metric_errors=metric_errors, prediction_days=prediction_days
            )
            
            if top_predictions is not None:
                # Format the numeric columns
                top_predictions['predicted_value'] = top_predictions['predicted_value'].round(2)
                top_predictions['confidence'] = top_predictions['confidence'].round(4)
                
                # Add a timestamp column for when the prediction was made
                top_predictions['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Add additional context about how to interpret the confidence value
                top_predictions['confidence_pct'] = (top_predictions['confidence'] * 100).round(1).astype(str) + '%'
                
                # Save to CSV
                csv_path = args.export_csv
                top_predictions.to_csv(csv_path, index=False)
                print(f"Exported top 50 predictions to {csv_path}")
                
                # Also print the top 10 most confident predictions
                print("\nTop 10 Most Confident Predictions Across All Stats:")
                print(top_predictions.head(10).to_string(index=False))
        
        # If --confident flag is set, show most confident predictions for each stat
        if args.confident:
            print("\n=== MOST CONFIDENT PREDICTIONS BY STATISTIC ===")
            
            for metric in TARGET_METRICS:
                if metric in models:
                    print(f"\nTop 10 Most Confident {metric.upper()} Predictions:")
                    confident_df = get_most_confident_predictions(
                        combined_data, models, metric, n=10, metric_errors=metric_errors
                    )
                    if confident_df is not None:
                        # Format the metric prediction to 2 decimal places
                        if metric in confident_df.columns:
                            confident_df[metric] = confident_df[metric].round(2)
                        
                        # Format confidence as percentage
                        conf_col = f"{metric}_confidence"
                        if conf_col in confident_df.columns:
                            confident_df[conf_col] = (confident_df[conf_col] * 100).round(1).astype(str) + '%'
                            
                        print(confident_df.to_string(index=False))
                    else:
                        print(f"No confident predictions available for {metric}")
        
        # If a specific player is specified, predict for that player
        if args.predict:
            print(f"\nPredicting performance for {args.predict}:")
            
            # Pass opponent if specified
            opponent = args.opponent
            if opponent:
                print(f"Using specified opponent: {opponent}")
                
            # Make the prediction
            pred_df = predict_player_performance(args.predict, models, combined_data, 
                                                metric_errors, opponent=opponent)
            
            if pred_df is not None:
                # Format confidence columns as percentages
                confidence_cols = []
                for col in pred_df.columns:
                    if col.endswith('_confidence'):
                        pred_df[col] = (pred_df[col] * 100).round(1).astype(str) + '%'
                        confidence_cols.append(col)
                
                # Round prediction values to 2 decimal places
                for metric in TARGET_METRICS:
                    if metric in pred_df.columns:
                        pred_df[metric] = pred_df[metric].round(2)
                
                # Include detailed confidence breakdown if requested
                if args.detailed_confidence:
                    # Show all confidence factors
                    detail_cols = [col for col in pred_df.columns 
                                 if any(x in col for x in ['_model_accuracy', '_player_consistency', 
                                                           '_recent_form', '_matchup', '_schedule'])]
                    
                    # Format these as percentages too
                    for col in detail_cols:
                        if col in pred_df.columns and pred_df[col].dtype != 'object':
                            pred_df[col] = (pred_df[col] * 100).round(1).astype(str) + '%'
                    
                    # Format trend columns
                    trend_cols = [col for col in pred_df.columns if 'trend_last_5' in col]
                    for col in trend_cols:
                        if col in pred_df.columns and pred_df[col].dtype != 'object':
                            # Add + for positive trends
                            value = float(pred_df[col].iloc[0])
                            if not pd.isna(value):
                                sign = '+' if value > 0 else ''
                                pred_df[col] = f"{sign}{value:.1f}%"
                    
                    # Re-order columns for better readability
                    display_cols = ['player_name', 'position', 'team', 'opponent']
                    
                    # Add each metric with its confidence score and factors
                    for metric in TARGET_METRICS:
                        if metric in pred_df.columns:
                            display_cols.append(metric)
                            metric_conf = f"{metric}_confidence"
                            if metric_conf in pred_df.columns:
                                display_cols.append(metric_conf)
                            
                            # Add confidence factors for this metric
                            for suffix in ['_model_accuracy', '_player_consistency', '_recent_form', 
                                          '_matchup', '_schedule']:
                                factor_col = f"{metric}{suffix}"
                                if factor_col in pred_df.columns:
                                    display_cols.append(factor_col)
                            
                            # Add trend for this metric
                            trend_col = f"{metric}_trend_last_5"
                            if trend_col in pred_df.columns:
                                display_cols.append(trend_col)
                    
                    # Add any remaining columns
                    for col in pred_df.columns:
                        if col not in display_cols:
                            display_cols.append(col)
                    
                    # Filter to columns that actually exist
                    display_cols = [col for col in display_cols if col in pred_df.columns]
                    
                    # Print with detailed breakdown
                    print("\nDetailed Prediction with Confidence Breakdown:")
                    print(pred_df[display_cols].to_string(index=False))
                
                else:
                    # Basic display - just show predictions and overall confidence
                    display_cols = ['player_name', 'position', 'team', 'opponent']
                    for metric in TARGET_METRICS:
                        if metric in pred_df.columns:
                            display_cols.append(metric)
                            metric_conf = f"{metric}_confidence"
                            if metric_conf in pred_df.columns:
                                display_cols.append(metric_conf)
                    
                    # Filter to columns that actually exist
                    display_cols = [col for col in display_cols if col in pred_df.columns]
                    
                    print(pred_df[display_cols].to_string(index=False))
        
        # If top N players are requested, predict for them
        if args.top_n:
            # Get top N players by fantasy points average
            top_players = train_data.groupby('name')['fantasy_points'].mean().nlargest(args.top_n).index
            print(f"\nPredicting performance for top {args.top_n} players by fantasy points:")
            
            # Make predictions for each top player
            all_preds = []
            for player in top_players:
                pred_df = predict_player_performance(player, models, combined_data, metric_errors)
                if pred_df is not None:
                    # Only keep prediction columns, not confidence
                    cols_to_keep = ['player_name', 'position', 'team', 'minutes_last_10'] + TARGET_METRICS
                    cols_to_keep = [col for col in cols_to_keep if col in pred_df.columns]
                    all_preds.append(pred_df[cols_to_keep])
            
            # Combine predictions
            if all_preds:
                all_preds_df = pd.concat(all_preds)
                # Round predictions to 2 decimal places
                for metric in TARGET_METRICS:
                    if metric in all_preds_df.columns:
                        all_preds_df[metric] = all_preds_df[metric].round(2)
                        
                print(all_preds_df.to_string(index=False))
        
        print("\nDone!")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()