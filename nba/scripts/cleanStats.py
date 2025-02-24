import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
import os

def connect_to_db():
    """Connect to the SQLite database."""
    return sqlite3.connect('../data/unified_nba_stats.db')

def load_data():
    """Load data from SQLite database into a pandas DataFrame."""
    conn = connect_to_db()
    query = "SELECT * FROM player_game_stats"  # Adjust table name if different
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def remove_duplicates(df):
    """Remove duplicate entries based on player, game, and date."""
    # Sort by date and keep the most recent entry in case of duplicates
    df = df.sort_values('game_date', ascending=False)
    df = df.drop_duplicates(subset=['player_id', 'game_id'], keep='first')
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Fill missing numeric stats with 0
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # For non-numeric columns, forward fill then backward fill
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric_columns] = df[non_numeric_columns].fillna(method='ffill').fillna(method='bfill')
    
    return df

def encode_categorical_features(df):
    """Encode categorical features into numeric format."""
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Binary encoding for home/away games
    if 'is_home' in df.columns:
        df['is_home'] = (df['is_home'] == 'home').astype(int)
    elif 'home_away' in df.columns:
        df['is_home'] = (df['home_away'] == 'home').astype(int)
    
    # Label encoding for team IDs
    le_team = LabelEncoder()
    if 'opponent_team_id' in df.columns:
        df['opponent_team_id_encoded'] = le_team.fit_transform(df['opponent_team_id'])
    
    # Create season and month features from game_date if available
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['season_month'] = df['game_date'].dt.month
        # Create season feature (e.g., 2023 for 2023-24 season)
        df['season'] = df['game_date'].apply(
            lambda x: x.year if x.month >= 8 else x.year - 1
        )
    
    # Handle any other categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        # Skip date columns and ID columns that should remain as is
        if col not in ['game_date'] and not col.endswith('_id'):
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    return df

def normalize_features(df):
    """Normalize features using appropriate scaling techniques."""
    # Create directory for scalers if it doesn't exist
    scaler_dir = '../models/scalers'
    os.makedirs(scaler_dir, exist_ok=True)
    
    # Create copy of dataframe
    df = df.copy()
    
    # Features to normalize with StandardScaler (for normal distributions)
    standard_scale_features = [
        'points', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers',
        'points_rolling_avg', 'rebounds_rolling_avg', 'assists_rolling_avg',
        'steals_rolling_avg', 'blocks_rolling_avg', 'turnovers_rolling_avg'
    ]
    
    # Features to normalize with MinMaxScaler (for bounded values)
    minmax_scale_features = [
        'minutes_played', 'minutes_played_rolling_avg',
        'is_home', 'season_month'
    ]
    
    # Initialize scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    
    # Apply StandardScaler to appropriate features
    standard_features = [col for col in standard_scale_features if col in df.columns]
    if standard_features:
        df[standard_features] = standard_scaler.fit_transform(df[standard_features])
        joblib.dump(standard_scaler, f'{scaler_dir}/standard_scaler.joblib')
    
    # Apply MinMaxScaler to appropriate features
    minmax_features = [col for col in minmax_scale_features if col in df.columns]
    if minmax_features:
        df[minmax_features] = minmax_scaler.fit_transform(df[minmax_features])
        joblib.dump(minmax_scaler, f'{scaler_dir}/minmax_scaler.joblib')
    
    # Special handling for odds-related features if they exist
    odds_features = [col for col in df.columns if 'odds' in col.lower()]
    if odds_features:
        odds_scaler = MinMaxScaler()
        df[odds_features] = odds_scaler.fit_transform(df[odds_features])
        joblib.dump(odds_scaler, f'{scaler_dir}/odds_scaler.joblib')
    
    return df

def calculate_team_stats(df):
    """Calculate aggregated team statistics."""
    if 'opponent_team_id' in df.columns and 'game_date' in df.columns:
        # Calculate rolling team averages
        team_stats = ['points_allowed', 'rebounds_allowed', 'assists_allowed']
        
        # Group by opponent and date to get team stats
        team_df = df.groupby(['opponent_team_id', 'game_date']).agg({
            'points': 'mean',
            'rebounds': 'mean',
            'assists': 'mean'
        }).reset_index()
        
        # Rename columns for clarity
        team_df.columns = ['team_id', 'game_date', 
                          'points_allowed', 'rebounds_allowed', 'assists_allowed']
        
        # Calculate rolling averages for team stats
        window = 5
        team_df = team_df.sort_values('game_date')
        for stat in team_stats:
            team_df[f'{stat}_rolling_avg'] = team_df.groupby('team_id')[stat].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Merge team stats back to original dataframe
        df = pd.merge(
            df,
            team_df[['team_id', 'game_date'] + [f'{stat}_rolling_avg' for stat in team_stats]],
            left_on=['opponent_team_id', 'game_date'],
            right_on=['team_id', 'game_date'],
            how='left'
        )
        
        df.drop('team_id', axis=1, inplace=True)
    
    return df

def calculate_rolling_averages(df, window=5):
    """Calculate rolling averages for key statistics."""
    # Sort by player and date
    df = df.sort_values(['player_id', 'game_date'])
    
    # List of columns to calculate rolling averages for
    stats_columns = ['points', 'rebounds', 'assists', 'steals', 'blocks', 
                    'turnovers', 'minutes_played']
    
    # Calculate rolling averages for each player
    for stat in stats_columns:
        if stat in df.columns:
            df[f'{stat}_rolling_avg'] = df.groupby('player_id')[stat].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    return df

def clean_obvious_errors(df):
    """Clean obvious errors in the data."""
    # Remove rows where minutes_played is unreasonably high (> 60 minutes)
    if 'minutes_played' in df.columns:
        df = df[df['minutes_played'] <= 60]
    
    # Remove rows where points/rebounds/assists are unreasonably high
    if 'points' in df.columns:
        df = df[df['points'] <= 100]
    if 'rebounds' in df.columns:
        df = df[df['rebounds'] <= 40]
    if 'assists' in df.columns:
        df = df[df['assists'] <= 30]
    
    return df

def save_cleaned_data(df):
    """Save the cleaned data back to the database."""
    conn = connect_to_db()
    # Create a new table for cleaned data
    df.to_sql('player_game_stats_cleaned', conn, if_exists='replace', index=False)
    conn.close()

def main():
    # Load data
    print("Loading data from database...")
    df = load_data()
    
    # Clean data
    print("Removing duplicates...")
    df = remove_duplicates(df)
    
    print("Cleaning obvious errors...")
    df = clean_obvious_errors(df)
    
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    print("Encoding categorical features...")
    df = encode_categorical_features(df)
    
    print("Calculating team statistics...")
    df = calculate_team_stats(df)
    
    print("Calculating rolling averages...")
    df = calculate_rolling_averages(df)
    
    print("Normalizing features...")
    df = normalize_features(df)
    
    # Save cleaned data
    print("Saving cleaned data...")
    save_cleaned_data(df)
    print("Data cleaning completed successfully!")

if __name__ == "__main__":
    main()